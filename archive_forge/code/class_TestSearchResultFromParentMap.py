from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
class TestSearchResultFromParentMap(TestGraphBase):

    def assertSearchResult(self, start_keys, stop_keys, key_count, parent_map, missing_keys=()):
        start, stop, count = vf_search.search_result_from_parent_map(parent_map, missing_keys)
        self.assertEqual((sorted(start_keys), sorted(stop_keys), key_count), (sorted(start), sorted(stop), count))

    def test_no_parents(self):
        self.assertSearchResult([], [], 0, {})
        self.assertSearchResult([], [], 0, None)

    def test_ancestry_1(self):
        self.assertSearchResult([b'rev4'], [NULL_REVISION], len(ancestry_1), ancestry_1)

    def test_ancestry_2(self):
        self.assertSearchResult([b'rev1b', b'rev4a'], [NULL_REVISION], len(ancestry_2), ancestry_2)
        self.assertSearchResult([b'rev1b', b'rev4a'], [], len(ancestry_2) + 1, ancestry_2, missing_keys=[NULL_REVISION])

    def test_partial_search(self):
        parent_map = {k: extended_history_shortcut[k] for k in [b'e', b'f']}
        self.assertSearchResult([b'e', b'f'], [b'd', b'a'], 2, parent_map)
        parent_map.update(((k, extended_history_shortcut[k]) for k in [b'd', b'a']))
        self.assertSearchResult([b'e', b'f'], [b'c', NULL_REVISION], 4, parent_map)
        parent_map[b'c'] = extended_history_shortcut[b'c']
        self.assertSearchResult([b'e', b'f'], [b'b'], 6, parent_map, missing_keys=[NULL_REVISION])
        parent_map[b'b'] = extended_history_shortcut[b'b']
        self.assertSearchResult([b'e', b'f'], [], 7, parent_map, missing_keys=[NULL_REVISION])