import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
class TreeTransform:
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

    trans_ids are only valid for the TreeTransform that generated them.
    """

    def __init__(self, tree, pb=None):
        self._tree = tree
        self._pb = pb
        self._id_number = 0
        self._tree_path_ids = {}
        self._tree_id_paths = {}
        self._new_name = {}
        self._new_parent = {}
        self._new_contents = {}
        self._removed_contents = set()
        self._new_executability = {}
        self._new_reference_revision = {}
        self._removed_id = set()
        self._done = False

    def __enter__(self):
        """Support Context Manager API."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support Context Manager API."""
        self.finalize()

    def iter_tree_children(self, trans_id):
        """Iterate through the entry's tree children, if any.

        :param trans_id: trans id to iterate
        :returns: Iterator over paths
        """
        raise NotImplementedError(self.iter_tree_children)

    def canonical_path(self, path):
        return path

    def tree_kind(self, trans_id):
        raise NotImplementedError(self.tree_kind)

    def by_parent(self):
        """Return a map of parent: children for known parents.

        Only new paths and parents of tree files with assigned ids are used.
        """
        by_parent = {}
        items = list(self._new_parent.items())
        items.extend(((t, self.final_parent(t)) for t in list(self._tree_id_paths)))
        for trans_id, parent_id in items:
            if parent_id not in by_parent:
                by_parent[parent_id] = set()
            by_parent[parent_id].add(trans_id)
        return by_parent

    def finalize(self):
        """Release the working tree lock, if held.

        This is required if apply has not been invoked, but can be invoked
        even after apply.
        """
        raise NotImplementedError(self.finalize)

    def create_path(self, name, parent):
        """Assign a transaction id to a new path"""
        trans_id = self.assign_id()
        unique_add(self._new_name, trans_id, name)
        unique_add(self._new_parent, trans_id, parent)
        return trans_id

    def adjust_path(self, name, parent, trans_id):
        """Change the path that is assigned to a transaction id."""
        if parent is None:
            raise ValueError('Parent trans-id may not be None')
        if trans_id == self.root:
            raise CantMoveRoot
        self._new_name[trans_id] = name
        self._new_parent[trans_id] = parent

    def adjust_root_path(self, name, parent):
        """Emulate moving the root by moving all children, instead.

        We do this by undoing the association of root's transaction id with the
        current tree.  This allows us to create a new directory with that
        transaction id.  We unversion the root directory and version the
        physically new directory, and hope someone versions the tree root
        later.
        """
        raise NotImplementedError(self.adjust_root_path)

    def fixup_new_roots(self):
        """Reinterpret requests to change the root directory

        Instead of creating a root directory, or moving an existing directory,
        all the attributes and children of the new root are applied to the
        existing root directory.

        This means that the old root trans-id becomes obsolete, so it is
        recommended only to invoke this after the root trans-id has become
        irrelevant.
        """
        raise NotImplementedError(self.fixup_new_roots)

    def assign_id(self):
        """Produce a new tranform id"""
        new_id = 'new-%s' % self._id_number
        self._id_number += 1
        return new_id

    def trans_id_tree_path(self, path):
        """Determine (and maybe set) the transaction ID for a tree path."""
        path = self.canonical_path(path)
        if path not in self._tree_path_ids:
            self._tree_path_ids[path] = self.assign_id()
            self._tree_id_paths[self._tree_path_ids[path]] = path
        return self._tree_path_ids[path]

    def get_tree_parent(self, trans_id):
        """Determine id of the parent in the tree."""
        path = self._tree_id_paths[trans_id]
        if path == '':
            return ROOT_PARENT
        return self.trans_id_tree_path(os.path.dirname(path))

    def delete_contents(self, trans_id):
        """Schedule the contents of a path entry for deletion"""
        kind = self.tree_kind(trans_id)
        if kind is not None:
            self._removed_contents.add(trans_id)

    def cancel_deletion(self, trans_id):
        """Cancel a scheduled deletion"""
        self._removed_contents.remove(trans_id)

    def delete_versioned(self, trans_id):
        """Delete and unversion a versioned file"""
        self.delete_contents(trans_id)
        self.unversion_file(trans_id)

    def set_executability(self, executability, trans_id):
        """Schedule setting of the 'execute' bit
        To unschedule, set to None
        """
        if executability is None:
            del self._new_executability[trans_id]
        else:
            unique_add(self._new_executability, trans_id, executability)

    def set_tree_reference(self, revision_id, trans_id):
        """Set the reference associated with a directory"""
        unique_add(self._new_reference_revision, trans_id, revision_id)

    def version_file(self, trans_id, file_id=None):
        """Schedule a file to become versioned."""
        raise NotImplementedError(self.version_file)

    def cancel_versioning(self, trans_id):
        """Undo a previous versioning of a file"""
        raise NotImplementedError(self.cancel_versioning)

    def unversion_file(self, trans_id):
        """Schedule a path entry to become unversioned"""
        self._removed_id.add(trans_id)

    def new_paths(self, filesystem_only=False):
        """Determine the paths of all new and changed files.

        :param filesystem_only: if True, only calculate values for files
            that require renames or execute bit changes.
        """
        raise NotImplementedError(self.new_paths)

    def final_kind(self, trans_id):
        """Determine the final file kind, after any changes applied.

        :return: None if the file does not exist/has no contents.  (It is
            conceivable that a path would be created without the corresponding
            contents insertion command)
        """
        if trans_id in self._new_contents:
            if trans_id in self._new_reference_revision:
                return 'tree-reference'
            return self._new_contents[trans_id]
        elif trans_id in self._removed_contents:
            return None
        else:
            return self.tree_kind(trans_id)

    def tree_path(self, trans_id):
        """Determine the tree path associated with the trans_id."""
        return self._tree_id_paths.get(trans_id)

    def final_is_versioned(self, trans_id):
        raise NotImplementedError(self.final_is_versioned)

    def final_parent(self, trans_id):
        """Determine the parent file_id, after any changes are applied.

        ROOT_PARENT is returned for the tree root.
        """
        try:
            return self._new_parent[trans_id]
        except KeyError:
            return self.get_tree_parent(trans_id)

    def final_name(self, trans_id):
        """Determine the final filename, after all changes are applied."""
        try:
            return self._new_name[trans_id]
        except KeyError:
            try:
                return os.path.basename(self._tree_id_paths[trans_id])
            except KeyError:
                raise NoFinalPath(trans_id, self)

    def path_changed(self, trans_id):
        """Return True if a trans_id's path has changed."""
        return trans_id in self._new_name or trans_id in self._new_parent

    def new_contents(self, trans_id):
        return trans_id in self._new_contents

    def find_raw_conflicts(self):
        """Find any violations of inventory or filesystem invariants"""
        raise NotImplementedError(self.find_raw_conflicts)

    def new_file(self, name, parent_id, contents, file_id=None, executable=None, sha1=None):
        """Convenience method to create files.

        name is the name of the file to create.
        parent_id is the transaction id of the parent directory of the file.
        contents is an iterator of bytestrings, which will be used to produce
        the file.
        :param file_id: The inventory ID of the file, if it is to be versioned.
        :param executable: Only valid when a file_id has been supplied.
        """
        raise NotImplementedError(self.new_file)

    def new_directory(self, name, parent_id, file_id=None):
        """Convenience method to create directories.

        name is the name of the directory to create.
        parent_id is the transaction id of the parent directory of the
        directory.
        file_id is the inventory ID of the directory, if it is to be versioned.
        """
        raise NotImplementedError(self.new_directory)

    def new_symlink(self, name, parent_id, target, file_id=None):
        """Convenience method to create symbolic link.

        name is the name of the symlink to create.
        parent_id is the transaction id of the parent directory of the symlink.
        target is a bytestring of the target of the symlink.
        file_id is the inventory ID of the file, if it is to be versioned.
        """
        raise NotImplementedError(self.new_symlink)

    def new_orphan(self, trans_id, parent_id):
        """Schedule an item to be orphaned.

        When a directory is about to be removed, its children, if they are not
        versioned are moved out of the way: they don't have a parent anymore.

        :param trans_id: The trans_id of the existing item.
        :param parent_id: The parent trans_id of the item.
        """
        raise NotImplementedError(self.new_orphan)

    def iter_changes(self):
        """Produce output in the same format as Tree.iter_changes.

        Will produce nonsensical results if invoked while inventory/filesystem
        conflicts (as reported by TreeTransform.find_raw_conflicts()) are present.

        This reads the Transform, but only reproduces changes involving a
        file_id.  Files that are not versioned in either of the FROM or TO
        states are not reflected.
        """
        raise NotImplementedError(self.iter_changes)

    def get_preview_tree(self):
        """Return a tree representing the result of the transform.

        The tree is a snapshot, and altering the TreeTransform will invalidate
        it.
        """
        raise NotImplementedError(self.get_preview_tree)

    def commit(self, branch, message, merge_parents=None, strict=False, timestamp=None, timezone=None, committer=None, authors=None, revprops=None, revision_id=None):
        """Commit the result of this TreeTransform to a branch.

        :param branch: The branch to commit to.
        :param message: The message to attach to the commit.
        :param merge_parents: Additional parent revision-ids specified by
            pending merges.
        :param strict: If True, abort the commit if there are unversioned
            files.
        :param timestamp: if not None, seconds-since-epoch for the time and
            date.  (May be a float.)
        :param timezone: Optional timezone for timestamp, as an offset in
            seconds.
        :param committer: Optional committer in email-id format.
            (e.g. "J Random Hacker <jrandom@example.com>")
        :param authors: Optional list of authors in email-id format.
        :param revprops: Optional dictionary of revision properties.
        :param revision_id: Optional revision id.  (Specifying a revision-id
            may reduce performance for some non-native formats.)
        :return: The revision_id of the revision committed.
        """
        raise NotImplementedError(self.commit)

    def create_file(self, contents, trans_id, mode_id=None, sha1=None):
        """Schedule creation of a new file.

        :seealso: new_file.

        :param contents: an iterator of strings, all of which will be written
            to the target destination.
        :param trans_id: TreeTransform handle
        :param mode_id: If not None, force the mode of the target file to match
            the mode of the object referenced by mode_id.
            Otherwise, we will try to preserve mode bits of an existing file.
        :param sha1: If the sha1 of this content is already known, pass it in.
            We can use it to prevent future sha1 computations.
        """
        raise NotImplementedError(self.create_file)

    def create_directory(self, trans_id):
        """Schedule creation of a new directory.

        See also new_directory.
        """
        raise NotImplementedError(self.create_directory)

    def create_symlink(self, target, trans_id):
        """Schedule creation of a new symbolic link.

        target is a bytestring.
        See also new_symlink.
        """
        raise NotImplementedError(self.create_symlink)

    def create_tree_reference(self, reference_revision, trans_id):
        raise NotImplementedError(self.create_tree_reference)

    def create_hardlink(self, path, trans_id):
        """Schedule creation of a hard link"""
        raise NotImplementedError(self.create_hardlink)

    def cancel_creation(self, trans_id):
        """Cancel the creation of new file contents."""
        raise NotImplementedError(self.cancel_creation)

    def cook_conflicts(self, raw_conflicts):
        """Cook conflicts.
        """
        raise NotImplementedError(self.cook_conflicts)