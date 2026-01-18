import pytest
import networkx as nx
class AbstractSNAP:
    node_attributes = ('color',)

    def build_original_graph(self):
        pass

    def build_summary_graph(self):
        pass

    def test_summary_graph(self):
        original_graph = self.build_original_graph()
        summary_graph = self.build_summary_graph()
        relationship_attributes = ('type',)
        generated_summary_graph = nx.snap_aggregation(original_graph, self.node_attributes, relationship_attributes)
        relabeled_summary_graph = self.deterministic_labels(generated_summary_graph)
        assert nx.is_isomorphic(summary_graph, relabeled_summary_graph)

    def deterministic_labels(self, G):
        node_labels = list(G.nodes)
        node_labels = sorted(node_labels, key=lambda n: sorted(G.nodes[n]['group'])[0])
        node_labels.sort()
        label_mapping = {}
        for index, node in enumerate(node_labels):
            label = 'Supernode-%s' % index
            label_mapping[node] = label
        return nx.relabel_nodes(G, label_mapping)