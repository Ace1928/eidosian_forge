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
class _PlanMerge(_PlanMergeBase):
    """Plan an annotate merge using on-the-fly annotation"""

    def __init__(self, a_rev, b_rev, vf, key_prefix):
        super().__init__(a_rev, b_rev, vf, key_prefix)
        self.a_key = self._key_prefix + (self.a_rev,)
        self.b_key = self._key_prefix + (self.b_rev,)
        self.graph = _mod_graph.Graph(self.vf)
        heads = self.graph.heads((self.a_key, self.b_key))
        if len(heads) == 1:
            self._head_key = heads.pop()
            if self._head_key == self.a_key:
                other = b_rev
            else:
                other = a_rev
            trace.mutter('found dominating revision for %s\n%s > %s', self.vf, self._head_key[-1], other)
            self._weave = None
        else:
            self._head_key = None
            self._build_weave()

    def _precache_tip_lines(self):
        pass

    def _find_recursive_lcas(self):
        """Find all the ancestors back to a unique lca"""
        cur_ancestors = (self.a_key, self.b_key)
        parent_map = {}
        while True:
            next_lcas = self.graph.find_lca(*cur_ancestors)
            if next_lcas == {_mod_revision.NULL_REVISION}:
                next_lcas = ()
            for rev_key in cur_ancestors:
                ordered_parents = tuple(self.graph.find_merge_order(rev_key, next_lcas))
                parent_map[rev_key] = ordered_parents
            if len(next_lcas) == 0:
                break
            elif len(next_lcas) == 1:
                parent_map[list(next_lcas)[0]] = ()
                break
            elif len(next_lcas) > 2:
                trace.mutter('More than 2 LCAs, falling back to all nodes for: %s, %s\n=> %s', self.a_key, self.b_key, cur_ancestors)
                cur_lcas = next_lcas
                while len(cur_lcas) > 1:
                    cur_lcas = self.graph.find_lca(*cur_lcas)
                if len(cur_lcas) == 0:
                    unique_lca = None
                else:
                    unique_lca = list(cur_lcas)[0]
                    if unique_lca == _mod_revision.NULL_REVISION:
                        unique_lca = None
                parent_map.update(self._find_unique_parents(next_lcas, unique_lca))
                break
            cur_ancestors = next_lcas
        return parent_map

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

    @staticmethod
    def _remove_external_references(parent_map):
        """Remove references that go outside of the parent map.

        :param parent_map: Something returned from Graph.get_parent_map(keys)
        :return: (filtered_parent_map, child_map, tails)
            filtered_parent_map is parent_map without external references
            child_map is the {parent_key: [child_keys]} mapping
            tails is a list of nodes that do not have any parents in the map
        """
        filtered_parent_map = {}
        child_map = {}
        tails = []
        for key, parent_keys in parent_map.items():
            culled_parent_keys = [p for p in parent_keys if p in parent_map]
            if not culled_parent_keys:
                tails.append(key)
            for parent_key in culled_parent_keys:
                child_map.setdefault(parent_key, []).append(key)
            child_map.setdefault(key, [])
            filtered_parent_map[key] = culled_parent_keys
        return (filtered_parent_map, child_map, tails)

    @staticmethod
    def _prune_tails(parent_map, child_map, tails_to_remove):
        """Remove tails from the parent map.

        This will remove the supplied revisions until no more children have 0
        parents.

        :param parent_map: A dict of {child: [parents]}, this dictionary will
            be modified in place.
        :param tails_to_remove: A list of tips that should be removed,
            this list will be consumed
        :param child_map: The reverse dict of parent_map ({parent: [children]})
            this dict will be modified
        :return: None, parent_map will be modified in place.
        """
        while tails_to_remove:
            next = tails_to_remove.pop()
            parent_map.pop(next)
            children = child_map.pop(next)
            for child in children:
                child_parents = parent_map[child]
                child_parents.remove(next)
                if len(child_parents) == 0:
                    tails_to_remove.append(child)

    def _get_interesting_texts(self, parent_map):
        """Return a dict of texts we are interested in.

        Note that the input is in key tuples, but the output is in plain
        revision ids.

        :param parent_map: The output from _find_recursive_lcas
        :return: A dict of {'revision_id':lines} as returned by
            _PlanMergeBase.get_lines()
        """
        all_revision_keys = set(parent_map)
        all_revision_keys.add(self.a_key)
        all_revision_keys.add(self.b_key)
        all_texts = self.get_lines([k[-1] for k in all_revision_keys])
        return all_texts

    def _build_weave(self):
        from .bzr import weave
        from .tsort import merge_sort
        self._weave = weave.Weave(weave_name='in_memory_weave', allow_reserved=True)
        parent_map = self._find_recursive_lcas()
        all_texts = self._get_interesting_texts(parent_map)
        tip_key = self._key_prefix + (_mod_revision.CURRENT_REVISION,)
        parent_map[tip_key] = (self.a_key, self.b_key)
        for seq_num, key, depth, eom in reversed(merge_sort(parent_map, tip_key)):
            if key == tip_key:
                continue
            parent_keys = parent_map[key]
            revision_id = key[-1]
            parent_ids = [k[-1] for k in parent_keys]
            self._weave.add_lines(revision_id, parent_ids, all_texts[revision_id])

    def plan_merge(self):
        """Generate a 'plan' for merging the two revisions.

        This involves comparing their texts and determining the cause of
        differences.  If text A has a line and text B does not, then either the
        line was added to text A, or it was deleted from B.  Once the causes
        are combined, they are written out in the format described in
        VersionedFile.plan_merge
        """
        if self._head_key is not None:
            if self._head_key == self.a_key:
                plan = 'new-a'
            else:
                if self._head_key != self.b_key:
                    raise AssertionError('There was an invalid head: %s != %s' % (self.b_key, self._head_key))
                plan = 'new-b'
            head_rev = self._head_key[-1]
            lines = self.get_lines([head_rev])[head_rev]
            return ((plan, line) for line in lines)
        return self._weave.plan_merge(self.a_rev, self.b_rev)