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
def _find_unique_parents(self, tip_keys, base_key):
    """Find ancestors of tip that aren't ancestors of base.

        :param tip_keys: Nodes that are interesting
        :param base_key: Cull all ancestors of this node
        :return: The parent map for all revisions between tip_keys and
            base_key. base_key will be included. References to nodes outside of
            the ancestor set will also be removed.
        """
    if base_key is None:
        parent_map = dict(self.graph.iter_ancestry(tip_keys))
        if _mod_revision.NULL_REVISION in parent_map:
            parent_map.pop(_mod_revision.NULL_REVISION)
    else:
        interesting = set()
        for tip in tip_keys:
            interesting.update(self.graph.find_unique_ancestors(tip, [base_key]))
        parent_map = self.graph.get_parent_map(interesting)
        parent_map[base_key] = ()
    culled_parent_map, child_map, tails = self._remove_external_references(parent_map)
    if base_key is not None:
        tails.remove(base_key)
        self._prune_tails(culled_parent_map, child_map, tails)
    simple_map = _mod_graph.collapse_linear_regions(culled_parent_map)
    return simple_map