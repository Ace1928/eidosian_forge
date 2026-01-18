import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
class TestGraphISOVF2pp:

    def test_custom_graph1_same_labels(self):
        G1 = nx.Graph()
        mapped = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'Z', 6: 'E'}
        edges1 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 6), (3, 4), (5, 1), (5, 2)]
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
        nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_edge(3, 7)
        G1.nodes[7]['label'] = 'blue'
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        G2.add_edges_from([(mapped[3], 'X'), (mapped[6], mapped[5])])
        G1.add_edge(4, 7)
        G2.nodes['X']['label'] = 'blue'
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.remove_edges_from([(1, 4), (1, 3)])
        G2.remove_edges_from([(mapped[1], mapped[5]), (mapped[1], mapped[2])])
        assert vf2pp_isomorphism(G1, G2, node_label='label')

    def test_custom_graph1_different_labels(self):
        G1 = nx.Graph()
        mapped = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'Z', 6: 'E'}
        edges1 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 6), (3, 4), (5, 1), (5, 2)]
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
        nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped

    def test_custom_graph2_same_labels(self):
        G1 = nx.Graph()
        mapped = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'G', 7: 'B', 6: 'F'}
        edges1 = [(1, 2), (1, 5), (5, 6), (2, 3), (2, 4), (3, 4), (4, 5), (2, 7)]
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
        nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G2.remove_edge(mapped[1], mapped[2])
        G2.add_edge(mapped[1], mapped[4])
        H1 = nx.Graph(G1.subgraph([2, 3, 4, 7]))
        H2 = nx.Graph(G2.subgraph([mapped[1], mapped[4], mapped[5], mapped[6]]))
        assert vf2pp_isomorphism(H1, H2, node_label='label')
        H1.add_edges_from([(3, 7), (4, 7)])
        H2.add_edges_from([(mapped[1], mapped[6]), (mapped[4], mapped[6])])
        assert vf2pp_isomorphism(H1, H2, node_label='label')

    def test_custom_graph2_different_labels(self):
        G1 = nx.Graph()
        mapped = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'G', 7: 'B', 6: 'F'}
        edges1 = [(1, 2), (1, 5), (5, 6), (2, 3), (2, 4), (3, 4), (4, 5), (2, 7)]
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
        nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
        G1.add_node(0)
        G2.add_node('Z')
        G1.nodes[0]['label'] = G1.nodes[1]['label']
        G2.nodes['Z']['label'] = G1.nodes[1]['label']
        mapped.update({0: 'Z'})
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
        G2.nodes['Z']['label'] = G1.nodes[2]['label']
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        G1.nodes[0]['label'] = 'blue'
        G2.nodes['Z']['label'] = 'blue'
        G1.add_edge(0, 1)
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        G2.add_edge('Z', 'A')
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped

    def test_custom_graph3_same_labels(self):
        G1 = nx.Graph()
        mapped = {1: 9, 2: 8, 3: 7, 4: 6, 5: 3, 8: 5, 9: 4, 7: 1, 6: 2}
        edges1 = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 7), (4, 9), (5, 8), (8, 9), (5, 6), (6, 7), (5, 2)]
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
        nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_edges_from([(6, 9), (7, 8)])
        G2.add_edges_from([(mapped[6], mapped[8]), (mapped[7], mapped[9])])
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        G1.add_edges_from([(6, 8), (7, 9)])
        G2.add_edges_from([(mapped[6], mapped[9]), (mapped[7], mapped[8])])
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_edges_from([(2, 7), (3, 6)])
        G2.add_edges_from([(mapped[2], mapped[7]), (mapped[3], mapped[6])])
        G1.add_node(10)
        G2.add_node('Z')
        G1.nodes[10]['label'] = 'blue'
        G2.nodes['Z']['label'] = 'blue'
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_edges_from([(10, 1), (10, 5), (10, 8)])
        G2.add_edges_from([('Z', mapped[1]), ('Z', mapped[4]), ('Z', mapped[9])])
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        H1 = nx.Graph(G1.subgraph([2, 3, 4, 5, 6, 7, 10]))
        H2 = nx.Graph(G2.subgraph([mapped[4], mapped[5], mapped[6], mapped[7], mapped[8], mapped[9], 'Z']))
        assert vf2pp_isomorphism(H1, H2, node_label='label') is None
        H1.add_edges_from([(10, 2), (10, 6), (3, 6), (2, 7), (2, 6), (3, 7)])
        H2.add_edges_from([('Z', mapped[7]), (mapped[6], mapped[9]), (mapped[7], mapped[8])])
        assert vf2pp_isomorphism(H1, H2, node_label='label')
        H1.add_edge(3, 5)
        H2.add_edge(mapped[5], mapped[7])
        assert vf2pp_isomorphism(H1, H2, node_label='label') is None

    def test_custom_graph3_different_labels(self):
        G1 = nx.Graph()
        mapped = {1: 9, 2: 8, 3: 7, 4: 6, 5: 3, 8: 5, 9: 4, 7: 1, 6: 2}
        edges1 = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 7), (4, 9), (5, 8), (8, 9), (5, 6), (6, 7), (5, 2)]
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
        nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
        G1.add_edge(1, 7)
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        G2.add_edge(9, 1)
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
        G1.add_node('A')
        G2.add_node('K')
        G1.nodes['A']['label'] = 'green'
        G2.nodes['K']['label'] = 'green'
        mapped.update({'A': 'K'})
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
        G1.add_edge('A', 6)
        G2.add_edge('K', 5)
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        G1.add_edge(1, 5)
        G1.add_edge(2, 9)
        G2.add_edge(9, 3)
        G2.add_edge(8, 4)
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        for node in G1.nodes():
            color = 'red'
            G1.nodes[node]['label'] = color
            G2.nodes[mapped[node]]['label'] = color
        assert vf2pp_isomorphism(G1, G2, node_label='label')

    def test_custom_graph4_different_labels(self):
        G1 = nx.Graph()
        edges1 = [(1, 2), (2, 3), (3, 8), (3, 4), (4, 5), (4, 6), (3, 6), (8, 7), (8, 9), (5, 9), (10, 11), (11, 12), (12, 13), (11, 13)]
        mapped = {1: 'n', 2: 'm', 3: 'l', 4: 'j', 5: 'k', 6: 'i', 7: 'g', 8: 'h', 9: 'f', 10: 'b', 11: 'a', 12: 'd', 13: 'e'}
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
        nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped

    def test_custom_graph4_same_labels(self):
        G1 = nx.Graph()
        edges1 = [(1, 2), (2, 3), (3, 8), (3, 4), (4, 5), (4, 6), (3, 6), (8, 7), (8, 9), (5, 9), (10, 11), (11, 12), (12, 13), (11, 13)]
        mapped = {1: 'n', 2: 'm', 3: 'l', 4: 'j', 5: 'k', 6: 'i', 7: 'g', 8: 'h', 9: 'f', 10: 'b', 11: 'a', 12: 'd', 13: 'e'}
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
        nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_node(0)
        G2.add_node('z')
        G1.nodes[0]['label'] = 'green'
        G2.nodes['z']['label'] = 'blue'
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        G2.nodes['z']['label'] = 'green'
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_edge(2, 5)
        G2.remove_edge('i', 'l')
        G2.add_edge('g', 'l')
        G2.add_edge('m', 'f')
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.remove_node(13)
        G2.remove_node('d')
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_edge(0, 10)
        G2.add_edge('e', 'z')
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_edge(11, 3)
        G1.add_edge(0, 8)
        G2.add_edge('a', 'l')
        G2.add_edge('z', 'j')
        assert vf2pp_isomorphism(G1, G2, node_label='label')

    def test_custom_graph5_same_labels(self):
        G1 = nx.Graph()
        edges1 = [(1, 5), (1, 2), (1, 4), (2, 3), (2, 6), (3, 4), (3, 7), (4, 8), (5, 8), (5, 6), (6, 7), (7, 8)]
        mapped = {1: 'a', 2: 'h', 3: 'd', 4: 'i', 5: 'g', 6: 'b', 7: 'j', 8: 'c'}
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
        nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        G1.add_edges_from([(3, 6), (2, 7), (2, 5), (1, 3), (4, 7), (6, 8)])
        G2.add_edges_from([(mapped[6], mapped[3]), (mapped[2], mapped[7]), (mapped[1], mapped[6]), (mapped[5], mapped[7]), (mapped[3], mapped[8]), (mapped[2], mapped[4])])
        assert vf2pp_isomorphism(G1, G2, node_label='label')
        H1 = nx.Graph(G1.subgraph([1, 5, 8, 6, 7, 3]))
        H2 = nx.Graph(G2.subgraph([mapped[1], mapped[4], mapped[8], mapped[7], mapped[3], mapped[5]]))
        assert vf2pp_isomorphism(H1, H2, node_label='label')
        H1.remove_node(8)
        H2.remove_node(mapped[7])
        assert vf2pp_isomorphism(H1, H2, node_label='label')
        H1.add_edge(1, 6)
        H1.remove_edge(3, 6)
        assert vf2pp_isomorphism(H1, H2, node_label='label')

    def test_custom_graph5_different_labels(self):
        G1 = nx.Graph()
        edges1 = [(1, 5), (1, 2), (1, 4), (2, 3), (2, 6), (3, 4), (3, 7), (4, 8), (5, 8), (5, 6), (6, 7), (7, 8)]
        mapped = {1: 'a', 2: 'h', 3: 'd', 4: 'i', 5: 'g', 6: 'b', 7: 'j', 8: 'c'}
        G1.add_edges_from(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        colors = ['red', 'blue', 'grey', 'none', 'brown', 'solarized', 'yellow', 'pink']
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
        nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
        c = 0
        for node in G1.nodes():
            color1 = colors[c]
            color2 = colors[(c + 3) % len(colors)]
            G1.nodes[node]['label'] = color1
            G2.nodes[mapped[node]]['label'] = color2
            c += 1
        assert vf2pp_isomorphism(G1, G2, node_label='label') is None
        H1 = G1.subgraph([1, 5])
        H2 = G2.subgraph(['i', 'c'])
        c = 0
        for node1, node2 in zip(H1.nodes(), H2.nodes()):
            H1.nodes[node1]['label'] = 'red'
            H2.nodes[node2]['label'] = 'red'
            c += 1
        assert vf2pp_isomorphism(H1, H2, node_label='label')

    def test_disconnected_graph_all_same_labels(self):
        G1 = nx.Graph()
        G1.add_nodes_from(list(range(10)))
        mapped = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
        nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label')

    def test_disconnected_graph_all_different_labels(self):
        G1 = nx.Graph()
        G1.add_nodes_from(list(range(10)))
        mapped = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}
        G2 = nx.relabel_nodes(G1, mapped)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
        nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped

    def test_disconnected_graph_some_same_labels(self):
        G1 = nx.Graph()
        G1.add_nodes_from(list(range(10)))
        mapped = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}
        G2 = nx.relabel_nodes(G1, mapped)
        colors = ['white', 'white', 'white', 'purple', 'purple', 'red', 'red', 'pink', 'pink', 'pink']
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(colors))), 'label')
        nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(colors))), 'label')
        assert vf2pp_isomorphism(G1, G2, node_label='label')