import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
class MergeIntoMerger(Merger):
    """Merger that understands other_tree will be merged into a subdir.

    This also changes the Merger api so that it uses real Branch, revision_id,
    and RevisonTree objects, rather than using revision specs.
    """

    def __init__(self, this_tree, other_branch, other_tree, target_subdir, source_subpath, other_rev_id=None):
        """Create a new MergeIntoMerger object.

        source_subpath in other_tree will be effectively copied to
        target_subdir in this_tree.

        :param this_tree: The tree that we will be merging into.
        :param other_branch: The Branch we will be merging from.
        :param other_tree: The RevisionTree object we want to merge.
        :param target_subdir: The relative path where we want to merge
            other_tree into this_tree
        :param source_subpath: The relative path specifying the subtree of
            other_tree to merge into this_tree.
        """
        null_ancestor_tree = this_tree.branch.repository.revision_tree(_mod_revision.NULL_REVISION)
        super().__init__(this_branch=this_tree.branch, this_tree=this_tree, other_tree=other_tree, base_tree=null_ancestor_tree)
        self._target_subdir = target_subdir
        self._source_subpath = source_subpath
        self.other_branch = other_branch
        if other_rev_id is None:
            other_rev_id = other_tree.get_revision_id()
        self.other_rev_id = self.other_basis = other_rev_id
        self.base_is_ancestor = True
        self.backup_files = True
        self.merge_type = Merge3Merger
        self.show_base = False
        self.reprocess = False
        self.interesting_files = None
        self.merge_type = _MergeTypeParameterizer(MergeIntoMergeType, target_subdir=self._target_subdir, source_subpath=self._source_subpath)
        if self._source_subpath != '':
            self._maybe_fetch(self.other_branch, self.this_branch, self.other_basis)

    def set_pending(self):
        if self._source_subpath != '':
            return
        Merger.set_pending(self)