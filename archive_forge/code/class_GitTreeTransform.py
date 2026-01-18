import errno
import os
import posixpath
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from dulwich.index import blob_from_path_and_stat, commit_tree
from dulwich.objects import Blob
from .. import annotate, conflicts, errors, multiparent, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import InterTree, TreeChange
from .mapping import (decode_git_path, encode_git_path, mode_is_executable,
from .tree import GitTree, GitTreeDirectory, GitTreeFile, GitTreeSymlink
class GitTreeTransform(DiskTreeTransform):
    """Represent a tree transformation.

    This object is designed to support incremental generation of the transform,
    in any order.

    However, it gives optimum performance when parent directories are created
    before their contents.  The transform is then able to put child files
    directly in their parent directory, avoiding later renames.

    It is easy to produce malformed transforms, but they are generally
    harmless.  Attempting to apply a malformed transform will cause an
    exception to be raised before any modifications are made to the tree.

    Many kinds of malformed transforms can be corrected with the
    resolve_conflicts function.  The remaining ones indicate programming error,
    such as trying to create a file with no path.

    Two sets of file creation methods are supplied.  Convenience methods are:
     * new_file
     * new_directory
     * new_symlink

    These are composed of the low-level methods:
     * create_path
     * create_file or create_directory or create_symlink
     * version_file
     * set_executability

    Transform/Transaction ids
    -------------------------
    trans_ids are temporary ids assigned to all files involved in a transform.
    It's possible, even common, that not all files in the Tree have trans_ids.

    trans_ids are used because filenames and file_ids are not good enough
    identifiers; filenames change.

    trans_ids are only valid for the TreeTransform that generated them.

    Limbo
    -----
    Limbo is a temporary directory use to hold new versions of files.
    Files are added to limbo by create_file, create_directory, create_symlink,
    and their convenience variants (new_*).  Files may be removed from limbo
    using cancel_creation.  Files are renamed from limbo into their final
    location as part of TreeTransform.apply

    Limbo must be cleaned up, by either calling TreeTransform.apply or
    calling TreeTransform.finalize.

    Files are placed into limbo inside their parent directories, where
    possible.  This reduces subsequent renames, and makes operations involving
    lots of files faster.  This optimization is only possible if the parent
    directory is created *before* creating any of its children, so avoid
    creating children before parents, where possible.

    Pending-deletion
    ----------------
    This temporary directory is used by _FileMover for storing files that are
    about to be deleted.  In case of rollback, the files will be restored.
    FileMover does not delete files until it is sure that a rollback will not
    happen.
    """

    def __init__(self, tree, pb=None):
        """Note: a tree_write lock is taken on the tree.

        Use TreeTransform.finalize() to release the lock (can be omitted if
        TreeTransform.apply() called).
        """
        tree.lock_tree_write()
        try:
            limbodir = urlutils.local_path_from_url(tree._transport.abspath('limbo'))
            osutils.ensure_empty_directory_exists(limbodir, errors.ExistingLimbo)
            deletiondir = urlutils.local_path_from_url(tree._transport.abspath('pending-deletion'))
            osutils.ensure_empty_directory_exists(deletiondir, errors.ExistingPendingDeletion)
        except BaseException:
            tree.unlock()
            raise
        self._realpaths = {}
        self._relpaths = {}
        DiskTreeTransform.__init__(self, tree, limbodir, pb, tree.case_sensitive)
        self._deletiondir = deletiondir

    def canonical_path(self, path):
        """Get the canonical tree-relative path"""
        abs = self._tree.abspath(path)
        if abs in self._relpaths:
            return self._relpaths[abs]
        dirname, basename = os.path.split(abs)
        if dirname not in self._realpaths:
            self._realpaths[dirname] = os.path.realpath(dirname)
        dirname = self._realpaths[dirname]
        abs = osutils.pathjoin(dirname, basename)
        if dirname in self._relpaths:
            relpath = osutils.pathjoin(self._relpaths[dirname], basename)
            relpath = relpath.rstrip('/\\')
        else:
            relpath = self._tree.relpath(abs)
        self._relpaths[abs] = relpath
        return relpath

    def tree_kind(self, trans_id):
        """Determine the file kind in the working tree.

        :returns: The file kind or None if the file does not exist
        """
        path = self._tree_id_paths.get(trans_id)
        if path is None:
            return None
        try:
            return osutils.file_kind(self._tree.abspath(path))
        except _mod_transport.NoSuchFile:
            return None

    def _set_mode(self, trans_id, mode_id, typefunc):
        """Set the mode of new file contents.
        The mode_id is the existing file to get the mode from (often the same
        as trans_id).  The operation is only performed if there's a mode match
        according to typefunc.
        """
        if mode_id is None:
            mode_id = trans_id
        try:
            old_path = self._tree_id_paths[mode_id]
        except KeyError:
            return
        try:
            mode = os.stat(self._tree.abspath(old_path)).st_mode
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.ENOTDIR):
                return
            else:
                raise
        if typefunc(mode):
            osutils.chmod_if_possible(self._limbo_name(trans_id), mode)

    def iter_tree_children(self, parent_id):
        """Iterate through the entry's tree children, if any"""
        try:
            path = self._tree_id_paths[parent_id]
        except KeyError:
            return
        try:
            children = os.listdir(self._tree.abspath(path))
        except (NotADirectoryError, FileNotFoundError):
            return
        for child in children:
            childpath = joinpath(path, child)
            if self._tree.is_control_filename(childpath):
                continue
            yield self.trans_id_tree_path(childpath)

    def _generate_limbo_path(self, trans_id):
        """Generate a limbo path using the final path if possible.

        This optimizes the performance of applying the tree transform by
        avoiding renames.  These renames can be avoided only when the parent
        directory is already scheduled for creation.

        If the final path cannot be used, falls back to using the trans_id as
        the relpath.
        """
        parent = self._new_parent.get(trans_id)
        use_direct_path = False
        if self._new_contents.get(parent) == 'directory':
            filename = self._new_name.get(trans_id)
            if filename is not None:
                if parent not in self._limbo_children:
                    self._limbo_children[parent] = set()
                    self._limbo_children_names[parent] = {}
                    use_direct_path = True
                elif self._case_sensitive_target:
                    if self._limbo_children_names[parent].get(filename) in (trans_id, None):
                        use_direct_path = True
                else:
                    for l_filename, l_trans_id in self._limbo_children_names[parent].items():
                        if l_trans_id == trans_id:
                            continue
                        if l_filename.lower() == filename.lower():
                            break
                    else:
                        use_direct_path = True
        if not use_direct_path:
            return DiskTreeTransform._generate_limbo_path(self, trans_id)
        limbo_name = osutils.pathjoin(self._limbo_files[parent], filename)
        self._limbo_children[parent].add(trans_id)
        self._limbo_children_names[parent][filename] = trans_id
        return limbo_name

    def cancel_versioning(self, trans_id):
        """Undo a previous versioning of a file"""
        self._versioned.remove(trans_id)

    def apply(self, no_conflicts=False, _mover=None):
        """Apply all changes to the inventory and filesystem.

        If filesystem or inventory conflicts are present, MalformedTransform
        will be thrown.

        If apply succeeds, finalize is not necessary.

        :param no_conflicts: if True, the caller guarantees there are no
            conflicts, so no check is made.
        :param _mover: Supply an alternate FileMover, for testing
        """
        for hook in MutableTree.hooks['pre_transform']:
            hook(self._tree, self)
        if not no_conflicts:
            self._check_malformed()
        self.rename_count = 0
        with ui.ui_factory.nested_progress_bar() as child_pb:
            child_pb.update(gettext('Apply phase'), 0, 2)
            index_changes = self._generate_index_changes()
            offset = 1
            if _mover is None:
                mover = _FileMover()
            else:
                mover = _mover
            try:
                child_pb.update(gettext('Apply phase'), 0 + offset, 2 + offset)
                self._apply_removals(mover)
                child_pb.update(gettext('Apply phase'), 1 + offset, 2 + offset)
                modified_paths = self._apply_insertions(mover)
            except BaseException:
                mover.rollback()
                raise
            else:
                mover.apply_deletions()
        self._tree._apply_index_changes(index_changes)
        self._done = True
        self.finalize()
        return _TransformResults(modified_paths, self.rename_count)

    def _apply_removals(self, mover):
        """Perform tree operations that remove directory/inventory names.

        That is, delete files that are to be deleted, and put any files that
        need renaming into limbo.  This must be done in strict child-to-parent
        order.

        If inventory_delta is None, no inventory delta generation is performed.
        """
        tree_paths = sorted(self._tree_path_ids.items(), reverse=True)
        with ui.ui_factory.nested_progress_bar() as child_pb:
            for num, (path, trans_id) in enumerate(tree_paths):
                if path == '':
                    continue
                child_pb.update(gettext('removing file'), num, len(tree_paths))
                full_path = self._tree.abspath(path)
                if trans_id in self._removed_contents:
                    delete_path = os.path.join(self._deletiondir, trans_id)
                    mover.pre_delete(full_path, delete_path)
                elif trans_id in self._new_name or trans_id in self._new_parent:
                    try:
                        mover.rename(full_path, self._limbo_name(trans_id))
                    except TransformRenameFailed as e:
                        if e.errno != errno.ENOENT:
                            raise
                    else:
                        self.rename_count += 1

    def _apply_insertions(self, mover):
        """Perform tree operations that insert directory/inventory names.

        That is, create any files that need to be created, and restore from
        limbo any files that needed renaming.  This must be done in strict
        parent-to-child order.

        If inventory_delta is None, no inventory delta is calculated, and
        no list of modified paths is returned.
        """
        new_paths = self.new_paths(filesystem_only=True)
        modified_paths = []
        with ui.ui_factory.nested_progress_bar() as child_pb:
            for num, (path, trans_id) in enumerate(new_paths):
                if num % 10 == 0:
                    child_pb.update(gettext('adding file'), num, len(new_paths))
                full_path = self._tree.abspath(path)
                if trans_id in self._needs_rename:
                    try:
                        mover.rename(self._limbo_name(trans_id), full_path)
                    except TransformRenameFailed as e:
                        if e.errno != errno.ENOENT:
                            raise
                    else:
                        self.rename_count += 1
                if trans_id in self._new_contents or self.path_changed(trans_id):
                    if trans_id in self._new_contents:
                        modified_paths.append(full_path)
                if trans_id in self._new_executability:
                    self._set_executability(path, trans_id)
                if trans_id in self._observed_sha1s:
                    o_sha1, o_st_val = self._observed_sha1s[trans_id]
                    st = osutils.lstat(full_path)
                    self._observed_sha1s[trans_id] = (o_sha1, st)
                if trans_id in self._new_reference_revision:
                    for submodule_path, submodule_url, submodule_name in self._tree._submodule_config():
                        if decode_git_path(submodule_path) == path:
                            break
                    else:
                        trace.warning('unable to find submodule for path %s', path)
                        continue
                    submodule_transport = self._tree.controldir.control_transport.clone(os.path.join('modules', submodule_name.decode('utf-8')))
                    submodule_transport.create_prefix()
                    from .dir import BareLocalGitControlDirFormat
                    BareLocalGitControlDirFormat().initialize_on_transport(submodule_transport)
                    with open(os.path.join(full_path, '.git'), 'w') as f:
                        submodule_abspath = submodule_transport.local_abspath('.')
                        f.write('gitdir: %s\n' % os.path.relpath(submodule_abspath, full_path))
        for path, trans_id in new_paths:
            if trans_id in self._limbo_files:
                del self._limbo_files[trans_id]
        self._new_contents.clear()
        return modified_paths

    def _generate_index_changes(self):
        """Generate an inventory delta for the current transform."""
        removed_id = set(self._removed_id)
        removed_id.update(self._removed_contents)
        changes = {}
        changed_ids = set()
        for id_set in [self._new_name, self._new_parent, self._new_executability, self._new_contents]:
            changed_ids.update(id_set)
        for id_set in [self._new_name, self._new_parent]:
            removed_id.update(id_set)
        changed_kind = set(self._new_contents)
        changed_kind.difference_update(changed_ids)
        changed_kind = (t for t in changed_kind if self.tree_kind(t) != self.final_kind(t))
        changed_ids.update(changed_kind)
        for t in changed_kind:
            if self.final_kind(t) == 'directory':
                removed_id.add(t)
                changed_ids.remove(t)
        new_paths = sorted(FinalPaths(self).get_paths(changed_ids))
        total_entries = len(new_paths) + len(removed_id)
        with ui.ui_factory.nested_progress_bar() as child_pb:
            for num, trans_id in enumerate(removed_id):
                if num % 10 == 0:
                    child_pb.update(gettext('removing file'), num, total_entries)
                try:
                    path = self._tree_id_paths[trans_id]
                except KeyError:
                    continue
                changes[path] = (None, None, None, None)
            for num, (path, trans_id) in enumerate(new_paths):
                if num % 10 == 0:
                    child_pb.update(gettext('adding file'), num + len(removed_id), total_entries)
                kind = self.final_kind(trans_id)
                if kind is None:
                    continue
                versioned = self.final_is_versioned(trans_id)
                if not versioned:
                    continue
                executability = self._new_executability.get(trans_id)
                reference_revision = self._new_reference_revision.get(trans_id)
                symlink_target = self._symlink_target.get(trans_id)
                changes[path] = (kind, executability, reference_revision, symlink_target)
        return [(p, k, e, rr, st) for p, (k, e, rr, st) in changes.items()]