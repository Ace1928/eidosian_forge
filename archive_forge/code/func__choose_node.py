import numbers
from collections import Counter
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import discrete_sequence, py_random_state, weighted_choice
def _choose_node(candidates, node_list, delta):
    if delta > 0:
        bias_sum = len(node_list) * delta
        p_delta = bias_sum / (bias_sum + len(candidates))
        if seed.random() < p_delta:
            return seed.choice(node_list)
    return seed.choice(candidates)