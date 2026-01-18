import os
import re
from collections import deque
from typing import TYPE_CHECKING, Optional, Type
from .. import branch as _mod_branch
from .. import controldir, debug, errors, lazy_import, osutils, revision, trace
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..mutabletree import MutableTree
from ..repository import Repository
from ..revisiontree import RevisionTree
from breezy import (
from breezy.bzr import (
from ..tree import (FileTimestampUnavailable, InterTree, MissingNestedTree,
class InventoryTree(Tree):
    """A tree that relies on an inventory for its metadata.

    Trees contain an `Inventory` object, and also know how to retrieve
    file texts mentioned in the inventory, either from a working
    directory or from a store.

    It is possible for trees to contain files that are not described
    in their inventory or vice versa; for this use `filenames()`.

    Subclasses should set the _inventory attribute, which is considered
    private to external API users.
    """

    def supports_symlinks(self):
        return True

    @classmethod
    def is_special_path(cls, path):
        return path.startswith('.bzr')

    def _get_root_inventory(self):
        return self._inventory
    root_inventory = property(_get_root_inventory, doc='Root inventory of this tree')
    supports_file_ids = True

    def _unpack_file_id(self, file_id):
        """Find the inventory and inventory file id for a tree file id.

        :param file_id: The tree file id, as bytestring or tuple
        :return: Inventory and inventory file id
        """
        if isinstance(file_id, tuple):
            if len(file_id) != 1:
                raise ValueError('nested trees not yet supported: %r' % file_id)
            file_id = file_id[0]
        return (self.root_inventory, file_id)

    def find_related_paths_across_trees(self, paths, trees=[], require_versioned=True):
        """Find related paths in tree corresponding to specified filenames in any
        of `lookup_trees`.

        All matches in all trees will be used, and all children of matched
        directories will be used.

        :param paths: The filenames to find related paths for (if None, returns
            None)
        :param trees: The trees to find file_ids within
        :param require_versioned: if true, all specified filenames must occur in
            at least one tree.
        :return: a set of paths for the specified filenames and their children
            in `tree`
        """
        if paths is None:
            return None
        file_ids = self.paths2ids(paths, trees, require_versioned=require_versioned)
        ret = set()
        for file_id in file_ids:
            try:
                ret.add(self.id2path(file_id))
            except errors.NoSuchId:
                pass
        return ret

    def paths2ids(self, paths, trees=[], require_versioned=True):
        """Return all the ids that can be reached by walking from paths.

        Each path is looked up in this tree and any extras provided in
        trees, and this is repeated recursively: the children in an extra tree
        of a directory that has been renamed under a provided path in this tree
        are all returned, even if none exist under a provided path in this
        tree, and vice versa.

        :param paths: An iterable of paths to start converting to ids from.
            Alternatively, if paths is None, no ids should be calculated and None
            will be returned. This is offered to make calling the api unconditional
            for code that *might* take a list of files.
        :param trees: Additional trees to consider.
        :param require_versioned: If False, do not raise NotVersionedError if
            an element of paths is not versioned in this tree and all of trees.
        """
        return find_ids_across_trees(paths, [self] + list(trees), require_versioned)

    def path2id(self, path):
        """Return the id for path in this tree."""
        with self.lock_read():
            return self._path2inv_file_id(path)[1]

    def is_versioned(self, path):
        return self.path2id(path) is not None

    def _path2ie(self, path):
        """Lookup an inventory entry by path.

        :param path: Path to look up
        :return: InventoryEntry
        """
        inv, ie = self._path2inv_ie(path)
        if ie is None:
            raise _mod_transport.NoSuchFile(path)
        return ie

    def _path2inv_ie(self, path):
        inv = self.root_inventory
        if isinstance(path, list):
            remaining = path
        else:
            remaining = osutils.splitpath(path)
        ie = inv.root
        while remaining:
            ie, base, remaining = inv.get_entry_by_path_partial(remaining)
            if remaining:
                inv = self._get_nested_tree('/'.join(base), ie.file_id, ie.reference_revision).root_inventory
        if ie is None:
            return (None, None)
        return (inv, ie)

    def _path2inv_file_id(self, path):
        """Lookup a inventory and inventory file id by path.

        :param path: Path to look up
        :return: tuple with inventory and inventory file id
        """
        inv, ie = self._path2inv_ie(path)
        if ie is None:
            return (None, None)
        return (inv, ie.file_id)

    def id2path(self, file_id, recurse='down'):
        """Return the path for a file id.

        :raises NoSuchId:
        """
        inventory, file_id = self._unpack_file_id(file_id)
        try:
            return inventory.id2path(file_id)
        except errors.NoSuchId:
            if recurse == 'down':
                if 'evil' in debug.debug_flags:
                    trace.mutter_callsite(2, 'id2path with nested trees scales with tree size.')
                for path in self.iter_references():
                    subtree = self.get_nested_tree(path)
                    try:
                        return osutils.pathjoin(path, subtree.id2path(file_id))
                    except errors.NoSuchId:
                        pass
            raise errors.NoSuchId(self, file_id)

    def all_file_ids(self):
        return {entry.file_id for path, entry in self.iter_entries_by_dir()}

    def all_versioned_paths(self):
        return {path for path, entry in self.iter_entries_by_dir()}

    def iter_entries_by_dir(self, specific_files=None, recurse_nested=False):
        """Walk the tree in 'by_dir' order.

        This will yield each entry in the tree as a (path, entry) tuple.
        The order that they are yielded is:

        See Tree.iter_entries_by_dir for details.
        """
        with self.lock_read():
            if specific_files is not None:
                inventory_file_ids = []
                for path in specific_files:
                    inventory, inv_file_id = self._path2inv_file_id(path)
                    if inventory and inventory is not self.root_inventory:
                        raise AssertionError('{!r} != {!r}'.format(inventory, self.root_inventory))
                    inventory_file_ids.append(inv_file_id)
            else:
                inventory_file_ids = None

            def iter_entries(inv):
                for p, e in inv.iter_entries_by_dir(specific_file_ids=inventory_file_ids):
                    if e.kind == 'tree-reference' and recurse_nested:
                        try:
                            subtree = self._get_nested_tree(p, e.file_id, e.reference_revision)
                        except errors.NotBranchError:
                            yield (p, e)
                        else:
                            with subtree.lock_read():
                                subinv = subtree.root_inventory
                                for subp, e in iter_entries(subinv):
                                    yield (osutils.pathjoin(p, subp) if subp else p, e)
                    else:
                        yield (p, e)
            return iter_entries(self.root_inventory)

    def iter_child_entries(self, path):
        with self.lock_read():
            ie = self._path2ie(path)
            if ie.kind != 'directory':
                raise errors.NotADirectory(path)
            return ie.children.values()

    def _get_plan_merge_data(self, path, other, base):
        from . import versionedfile
        file_id = self.path2id(path)
        vf = versionedfile._PlanMergeVersionedFile(file_id)
        last_revision_a = self._get_file_revision(path, file_id, vf, b'this:')
        last_revision_b = other._get_file_revision(other.id2path(file_id), file_id, vf, b'other:')
        if base is None:
            last_revision_base = None
        else:
            last_revision_base = base._get_file_revision(base.id2path(file_id), file_id, vf, b'base:')
        return (vf, last_revision_a, last_revision_b, last_revision_base)

    def plan_file_merge(self, path, other, base=None):
        """Generate a merge plan based on annotations.

        If the file contains uncommitted changes in this tree, they will be
        attributed to the 'current:' pseudo-revision.  If the file contains
        uncommitted changes in the other tree, they will be assigned to the
        'other:' pseudo-revision.
        """
        data = self._get_plan_merge_data(path, other, base)
        vf, last_revision_a, last_revision_b, last_revision_base = data
        return vf.plan_merge(last_revision_a, last_revision_b, last_revision_base)

    def plan_file_lca_merge(self, path, other, base=None):
        """Generate a merge plan based lca-newness.

        If the file contains uncommitted changes in this tree, they will be
        attributed to the 'current:' pseudo-revision.  If the file contains
        uncommitted changes in the other tree, they will be assigned to the
        'other:' pseudo-revision.
        """
        data = self._get_plan_merge_data(path, other, base)
        vf, last_revision_a, last_revision_b, last_revision_base = data
        return vf.plan_lca_merge(last_revision_a, last_revision_b, last_revision_base)

    def _iter_parent_trees(self):
        """Iterate through parent trees, defaulting to Tree.revision_tree."""
        for revision_id in self.get_parent_ids():
            try:
                yield self.revision_tree(revision_id)
            except errors.NoSuchRevisionInTree:
                yield self.branch.repository.revision_tree(revision_id)

    def _get_file_revision(self, path, file_id, vf, tree_revision):
        """Ensure that file_id, tree_revision is in vf to plan the merge."""
        from . import versionedfile
        last_revision = tree_revision
        parent_keys = [(file_id, t.get_file_revision(path)) for t in self._iter_parent_trees()]
        with self.get_file(path) as f:
            vf.add_content(versionedfile.FileContentFactory((file_id, last_revision), parent_keys, f, size=osutils.filesize(f)))
        repo = self.branch.repository
        base_vf = repo.texts
        if base_vf not in vf.fallback_versionedfiles:
            vf.fallback_versionedfiles.append(base_vf)
        return last_revision

    def preview_transform(self, pb=None):
        from .transform import TransformPreview
        return TransformPreview(self, pb=pb)