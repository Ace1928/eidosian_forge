import random
import networkx as nx
from networkx.algorithms.approximation import maxcut
def _cut_is_locally_optimal(G, cut_size, set1):
    for i, node in enumerate(set1):
        cut_size_without_node = nx.algorithms.cut_size(G, set1 - {node}, weight='weight')
        assert cut_size_without_node <= cut_size