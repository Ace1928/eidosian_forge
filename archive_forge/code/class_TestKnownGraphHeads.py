import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
class TestKnownGraphHeads(TestCaseWithKnownGraph):
    scenarios = caching_scenarios() + non_caching_scenarios()
    do_cache = None

    def test_heads_null(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertEqual({b'null:'}, graph.heads([b'null:']))
        self.assertEqual({b'rev1'}, graph.heads([b'null:', b'rev1']))
        self.assertEqual({b'rev1'}, graph.heads([b'rev1', b'null:']))
        self.assertEqual({b'rev1'}, graph.heads({b'rev1', b'null:'}))
        self.assertEqual({b'rev1'}, graph.heads((b'rev1', b'null:')))

    def test_heads_one(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertEqual({b'null:'}, graph.heads([b'null:']))
        self.assertEqual({b'rev1'}, graph.heads([b'rev1']))
        self.assertEqual({b'rev2a'}, graph.heads([b'rev2a']))
        self.assertEqual({b'rev2b'}, graph.heads([b'rev2b']))
        self.assertEqual({b'rev3'}, graph.heads([b'rev3']))
        self.assertEqual({b'rev4'}, graph.heads([b'rev4']))

    def test_heads_single(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertEqual({b'rev4'}, graph.heads([b'null:', b'rev4']))
        self.assertEqual({b'rev2a'}, graph.heads([b'rev1', b'rev2a']))
        self.assertEqual({b'rev2b'}, graph.heads([b'rev1', b'rev2b']))
        self.assertEqual({b'rev3'}, graph.heads([b'rev1', b'rev3']))
        self.assertEqual({b'rev3'}, graph.heads([b'rev3', b'rev2a']))
        self.assertEqual({b'rev4'}, graph.heads([b'rev1', b'rev4']))
        self.assertEqual({b'rev4'}, graph.heads([b'rev2a', b'rev4']))
        self.assertEqual({b'rev4'}, graph.heads([b'rev2b', b'rev4']))
        self.assertEqual({b'rev4'}, graph.heads([b'rev3', b'rev4']))

    def test_heads_two_heads(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertEqual({b'rev2a', b'rev2b'}, graph.heads([b'rev2a', b'rev2b']))
        self.assertEqual({b'rev3', b'rev2b'}, graph.heads([b'rev3', b'rev2b']))

    def test_heads_criss_cross(self):
        graph = self.make_known_graph(test_graph.criss_cross)
        self.assertEqual({b'rev2a'}, graph.heads([b'rev2a', b'rev1']))
        self.assertEqual({b'rev2b'}, graph.heads([b'rev2b', b'rev1']))
        self.assertEqual({b'rev3a'}, graph.heads([b'rev3a', b'rev1']))
        self.assertEqual({b'rev3b'}, graph.heads([b'rev3b', b'rev1']))
        self.assertEqual({b'rev2a', b'rev2b'}, graph.heads([b'rev2a', b'rev2b']))
        self.assertEqual({b'rev3a'}, graph.heads([b'rev3a', b'rev2a']))
        self.assertEqual({b'rev3a'}, graph.heads([b'rev3a', b'rev2b']))
        self.assertEqual({b'rev3a'}, graph.heads([b'rev3a', b'rev2a', b'rev2b']))
        self.assertEqual({b'rev3b'}, graph.heads([b'rev3b', b'rev2a']))
        self.assertEqual({b'rev3b'}, graph.heads([b'rev3b', b'rev2b']))
        self.assertEqual({b'rev3b'}, graph.heads([b'rev3b', b'rev2a', b'rev2b']))
        self.assertEqual({b'rev3a', b'rev3b'}, graph.heads([b'rev3a', b'rev3b']))
        self.assertEqual({b'rev3a', b'rev3b'}, graph.heads([b'rev3a', b'rev3b', b'rev2a', b'rev2b']))

    def test_heads_shortcut(self):
        graph = self.make_known_graph(test_graph.history_shortcut)
        self.assertEqual({b'rev2a', b'rev2b', b'rev2c'}, graph.heads([b'rev2a', b'rev2b', b'rev2c']))
        self.assertEqual({b'rev3a', b'rev3b'}, graph.heads([b'rev3a', b'rev3b']))
        self.assertEqual({b'rev3a', b'rev3b'}, graph.heads([b'rev2a', b'rev3a', b'rev3b']))
        self.assertEqual({b'rev2a', b'rev3b'}, graph.heads([b'rev2a', b'rev3b']))
        self.assertEqual({b'rev2c', b'rev3a'}, graph.heads([b'rev2c', b'rev3a']))

    def test_heads_linear(self):
        graph = self.make_known_graph(test_graph.racing_shortcuts)
        self.assertEqual({b'w'}, graph.heads([b'w', b's']))
        self.assertEqual({b'z'}, graph.heads([b'w', b's', b'z']))
        self.assertEqual({b'w', b'q'}, graph.heads([b'w', b's', b'q']))
        self.assertEqual({b'z'}, graph.heads([b's', b'z']))

    def test_heads_alt_merge(self):
        graph = self.make_known_graph(alt_merge)
        self.assertEqual({b'c'}, graph.heads([b'a', b'c']))

    def test_heads_with_ghost(self):
        graph = self.make_known_graph(test_graph.with_ghost)
        self.assertEqual({b'e', b'g'}, graph.heads([b'e', b'g']))
        self.assertEqual({b'a', b'c'}, graph.heads([b'a', b'c']))
        self.assertEqual({b'a', b'g'}, graph.heads([b'a', b'g']))
        self.assertEqual({b'f', b'g'}, graph.heads([b'f', b'g']))
        self.assertEqual({b'c'}, graph.heads([b'c', b'g']))
        self.assertEqual({b'c'}, graph.heads([b'c', b'b', b'd', b'g']))
        self.assertEqual({b'a', b'c'}, graph.heads([b'a', b'c', b'e', b'g']))
        self.assertEqual({b'a', b'c'}, graph.heads([b'a', b'c', b'f']))

    def test_filling_in_ghosts_resets_head_cache(self):
        graph = self.make_known_graph(test_graph.with_ghost)
        self.assertEqual({b'e', b'g'}, graph.heads([b'e', b'g']))
        graph.add_node(b'g', [b'e'])
        self.assertEqual({b'g'}, graph.heads([b'e', b'g']))