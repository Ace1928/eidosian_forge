import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def add_graphs(self, graph_list):
    """Add many graphs to this GraphML document."""
    for G in graph_list:
        self.add_graph_element(G)