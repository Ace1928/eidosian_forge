import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
class GraphTest(test.TestCase):

    def test_no_successors_no_predecessors(self):
        g = graph.DiGraph()
        g.add_node('a')
        g.add_node('b')
        g.add_node('c')
        g.add_edge('b', 'c')
        self.assertEqual(set(['a', 'b']), set(g.no_predecessors_iter()))
        self.assertEqual(set(['a', 'c']), set(g.no_successors_iter()))

    def test_directed(self):
        g = graph.DiGraph()
        g.add_node('a')
        g.add_node('b')
        g.add_edge('a', 'b')
        self.assertTrue(g.is_directed_acyclic())
        g.add_edge('b', 'a')
        self.assertFalse(g.is_directed_acyclic())

    def test_frozen(self):
        g = graph.DiGraph()
        self.assertFalse(g.frozen)
        g.add_node('b')
        g.freeze()
        self.assertRaises(nx.NetworkXError, g.add_node, 'c')

    def test_merge(self):
        g = graph.DiGraph()
        g.add_node('a')
        g.add_node('b')
        g2 = graph.DiGraph()
        g2.add_node('c')
        g3 = graph.merge_graphs(g, g2)
        self.assertEqual(3, len(g3))

    def test_pydot_output(self):
        for graph_cls, kind, edge in [(graph.OrderedDiGraph, 'digraph', '->'), (graph.OrderedGraph, 'graph', '--')]:
            g = graph_cls(name='test')
            g.add_node('a')
            g.add_node('b')
            g.add_node('c')
            g.add_edge('a', 'b')
            g.add_edge('b', 'c')
            expected = '\nstrict %(kind)s "test" {\na;\nb;\nc;\na %(edge)s b;\nb %(edge)s c;\n}\n' % {'kind': kind, 'edge': edge}
            self.assertEqual(expected.lstrip(), g.export_to_dot())

    def test_merge_edges(self):
        g = graph.DiGraph()
        g.add_node('a')
        g.add_node('b')
        g.add_edge('a', 'b')
        g2 = graph.DiGraph()
        g2.add_node('c')
        g2.add_node('d')
        g2.add_edge('c', 'd')
        g3 = graph.merge_graphs(g, g2)
        self.assertEqual(4, len(g3))
        self.assertTrue(g3.has_edge('c', 'd'))
        self.assertTrue(g3.has_edge('a', 'b'))

    def test_overlap_detector(self):
        g = graph.DiGraph()
        g.add_node('a')
        g.add_node('b')
        g.add_edge('a', 'b')
        g2 = graph.DiGraph()
        g2.add_node('a')
        g2.add_node('d')
        g2.add_edge('a', 'd')
        self.assertRaises(ValueError, graph.merge_graphs, g, g2)

        def occurrence_detector(to_graph, from_graph):
            return sum((1 for node in from_graph.nodes if node in to_graph))
        self.assertRaises(ValueError, graph.merge_graphs, g, g2, overlap_detector=occurrence_detector)
        g3 = graph.merge_graphs(g, g2, allow_overlaps=True)
        self.assertEqual(3, len(g3))
        self.assertTrue(g3.has_edge('a', 'b'))
        self.assertTrue(g3.has_edge('a', 'd'))

    def test_invalid_detector(self):
        g = graph.DiGraph()
        g.add_node('a')
        g2 = graph.DiGraph()
        g2.add_node('c')
        self.assertRaises(ValueError, graph.merge_graphs, g, g2, overlap_detector='b')