import collections
import networkx as nx
def _precheck_label_properties(graph_params):
    G1, G2, G1_labels, G2_labels, nodes_of_G1Labels, nodes_of_G2Labels, _ = graph_params
    if any((label not in nodes_of_G1Labels or len(nodes_of_G1Labels[label]) != len(nodes) for label, nodes in nodes_of_G2Labels.items())):
        return False
    return True