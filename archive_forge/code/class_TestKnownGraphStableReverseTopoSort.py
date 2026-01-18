import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
class TestKnownGraphStableReverseTopoSort(TestCaseWithKnownGraph):
    """Test the sort order returned by gc_sort."""

    def assertSorted(self, expected, parent_map):
        graph = self.make_known_graph(parent_map)
        value = graph.gc_sort()
        if expected != value:
            self.assertEqualDiff(pprint.pformat(expected), pprint.pformat(value))

    def test_empty(self):
        self.assertSorted([], {})

    def test_single(self):
        self.assertSorted(['a'], {'a': ()})
        self.assertSorted([('a',)], {('a',): ()})
        self.assertSorted([('F', 'a')], {('F', 'a'): ()})

    def test_linear(self):
        self.assertSorted(['c', 'b', 'a'], {'a': (), 'b': ('a',), 'c': ('b',)})
        self.assertSorted([('c',), ('b',), ('a',)], {('a',): (), ('b',): (('a',),), ('c',): (('b',),)})
        self.assertSorted([('F', 'c'), ('F', 'b'), ('F', 'a')], {('F', 'a'): (), ('F', 'b'): (('F', 'a'),), ('F', 'c'): (('F', 'b'),)})

    def test_mixed_ancestries(self):
        self.assertSorted([('F', 'c'), ('F', 'b'), ('F', 'a'), ('G', 'c'), ('G', 'b'), ('G', 'a'), ('Q', 'c'), ('Q', 'b'), ('Q', 'a')], {('F', 'a'): (), ('F', 'b'): (('F', 'a'),), ('F', 'c'): (('F', 'b'),), ('G', 'a'): (), ('G', 'b'): (('G', 'a'),), ('G', 'c'): (('G', 'b'),), ('Q', 'a'): (), ('Q', 'b'): (('Q', 'a'),), ('Q', 'c'): (('Q', 'b'),)})

    def test_stable_sorting(self):
        self.assertSorted(['b', 'c', 'a'], {'a': (), 'b': ('a',), 'c': ('a',)})
        self.assertSorted(['b', 'c', 'd', 'a'], {'a': (), 'b': ('a',), 'c': ('a',), 'd': ('a',)})
        self.assertSorted(['b', 'c', 'd', 'a'], {'a': (), 'b': ('a',), 'c': ('a',), 'd': ('a',)})
        self.assertSorted(['Z', 'b', 'c', 'd', 'a'], {'a': (), 'b': ('a',), 'c': ('a',), 'd': ('a',), 'Z': ('a',)})
        self.assertSorted(['e', 'b', 'c', 'f', 'Z', 'd', 'a'], {'a': (), 'b': ('a',), 'c': ('a',), 'd': ('a',), 'Z': ('a',), 'e': ('b', 'c', 'd'), 'f': ('d', 'Z')})

    def test_skip_ghost(self):
        self.assertSorted(['b', 'c', 'a'], {'a': (), 'b': ('a', 'ghost'), 'c': ('a',)})

    def test_skip_mainline_ghost(self):
        self.assertSorted(['b', 'c', 'a'], {'a': (), 'b': ('ghost', 'a'), 'c': ('a',)})