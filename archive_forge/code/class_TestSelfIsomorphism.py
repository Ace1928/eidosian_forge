import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
class TestSelfIsomorphism:
    data = [([(0, {'name': 'a'}), (1, {'name': 'a'}), (2, {'name': 'b'}), (3, {'name': 'b'}), (4, {'name': 'a'}), (5, {'name': 'a'})], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]), (range(1, 5), [(1, 2), (2, 4), (4, 3), (3, 1)]), ([], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (6, 7), (2, 8), (8, 9), (4, 10), (10, 11)]), ([], [(0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (3, 6)])]

    def test_self_isomorphism(self):
        """
        For some small, symmetric graphs, make sure that 1) they are isomorphic
        to themselves, and 2) that only the identity mapping is found.
        """
        for node_data, edge_data in self.data:
            graph = nx.Graph()
            graph.add_nodes_from(node_data)
            graph.add_edges_from(edge_data)
            ismags = iso.ISMAGS(graph, graph, node_match=iso.categorical_node_match('name', None))
            assert ismags.is_isomorphic()
            assert ismags.subgraph_is_isomorphic()
            assert list(ismags.subgraph_isomorphisms_iter(symmetry=True)) == [{n: n for n in graph.nodes}]

    def test_edgecase_self_isomorphism(self):
        """
        This edgecase is one of the cases in which it is hard to find all
        symmetry elements.
        """
        graph = nx.Graph()
        nx.add_path(graph, range(5))
        graph.add_edges_from([(2, 5), (5, 6)])
        ismags = iso.ISMAGS(graph, graph)
        ismags_answer = list(ismags.find_isomorphisms(True))
        assert ismags_answer == [{n: n for n in graph.nodes}]
        graph = nx.relabel_nodes(graph, {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 4, 6: 5})
        ismags = iso.ISMAGS(graph, graph)
        ismags_answer = list(ismags.find_isomorphisms(True))
        assert ismags_answer == [{n: n for n in graph.nodes}]

    def test_directed_self_isomorphism(self):
        """
        For some small, directed, symmetric graphs, make sure that 1) they are
        isomorphic to themselves, and 2) that only the identity mapping is
        found.
        """
        for node_data, edge_data in self.data:
            graph = nx.Graph()
            graph.add_nodes_from(node_data)
            graph.add_edges_from(edge_data)
            ismags = iso.ISMAGS(graph, graph, node_match=iso.categorical_node_match('name', None))
            assert ismags.is_isomorphic()
            assert ismags.subgraph_is_isomorphic()
            assert list(ismags.subgraph_isomorphisms_iter(symmetry=True)) == [{n: n for n in graph.nodes}]