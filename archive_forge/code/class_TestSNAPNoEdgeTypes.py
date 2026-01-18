import pytest
import networkx as nx
class TestSNAPNoEdgeTypes(AbstractSNAP):
    relationship_attributes = ()

    def test_summary_graph(self):
        original_graph = self.build_original_graph()
        summary_graph = self.build_summary_graph()
        relationship_attributes = ('type',)
        generated_summary_graph = nx.snap_aggregation(original_graph, self.node_attributes)
        relabeled_summary_graph = self.deterministic_labels(generated_summary_graph)
        assert nx.is_isomorphic(summary_graph, relabeled_summary_graph)

    def build_original_graph(self):
        nodes = {'A': {'color': 'Red'}, 'B': {'color': 'Red'}, 'C': {'color': 'Red'}, 'D': {'color': 'Red'}, 'E': {'color': 'Blue'}, 'F': {'color': 'Blue'}, 'G': {'color': 'Blue'}, 'H': {'color': 'Blue'}, 'I': {'color': 'Yellow'}, 'J': {'color': 'Yellow'}, 'K': {'color': 'Yellow'}, 'L': {'color': 'Yellow'}}
        edges = [('A', 'B'), ('A', 'C'), ('A', 'E'), ('A', 'I'), ('B', 'D'), ('B', 'J'), ('B', 'F'), ('C', 'G'), ('D', 'H'), ('I', 'J'), ('J', 'K'), ('I', 'L')]
        G = nx.Graph()
        for node in nodes:
            attributes = nodes[node]
            G.add_node(node, **attributes)
        for source, target in edges:
            G.add_edge(source, target)
        return G

    def build_summary_graph(self):
        nodes = {'Supernode-0': {'color': 'Red'}, 'Supernode-1': {'color': 'Red'}, 'Supernode-2': {'color': 'Blue'}, 'Supernode-3': {'color': 'Blue'}, 'Supernode-4': {'color': 'Yellow'}, 'Supernode-5': {'color': 'Yellow'}}
        edges = [('Supernode-0', 'Supernode-0'), ('Supernode-0', 'Supernode-1'), ('Supernode-0', 'Supernode-2'), ('Supernode-0', 'Supernode-4'), ('Supernode-1', 'Supernode-3'), ('Supernode-4', 'Supernode-4'), ('Supernode-4', 'Supernode-5')]
        G = nx.Graph()
        for node in nodes:
            attributes = nodes[node]
            G.add_node(node, **attributes)
        for source, target in edges:
            G.add_edge(source, target)
        supernodes = {'Supernode-0': {'A', 'B'}, 'Supernode-1': {'C', 'D'}, 'Supernode-2': {'E', 'F'}, 'Supernode-3': {'G', 'H'}, 'Supernode-4': {'I', 'J'}, 'Supernode-5': {'K', 'L'}}
        nx.set_node_attributes(G, supernodes, 'group')
        return G