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
def _do_merge_to(self):
    merge = self.make_merger()
    if self.other_branch is not None:
        self.other_branch.update_references(self.this_branch)
    for hook in Merger.hooks['pre_merge']:
        hook(merge)
    merge.do_merge()
    for hook in Merger.hooks['post_merge']:
        hook(merge)
    if self.recurse == 'down':
        for relpath in self.this_tree.iter_references():
            sub_tree = self.this_tree.get_nested_tree(relpath)
            other_revision = self.other_tree.get_reference_revision(relpath)
            if other_revision == sub_tree.last_revision():
                continue
            other_branch = self.other_tree.reference_parent(relpath)
            graph = self.this_tree.branch.repository.get_graph(other_branch.repository)
            if graph.is_ancestor(sub_tree.last_revision(), other_revision):
                sub_tree.pull(other_branch, stop_revision=other_revision)
            else:
                sub_merge = Merger(sub_tree.branch, this_tree=sub_tree)
                sub_merge.merge_type = self.merge_type
                sub_merge.set_other_revision(other_revision, other_branch)
                base_tree_path = _mod_tree.find_previous_path(self.this_tree, self.base_tree, relpath)
                if base_tree_path is None:
                    raise NotImplementedError
                base_revision = self.base_tree.get_reference_revision(base_tree_path)
                sub_merge.base_tree = sub_tree.branch.repository.revision_tree(base_revision)
                sub_merge.base_rev_id = base_revision
                sub_merge.do_merge()
    return merge