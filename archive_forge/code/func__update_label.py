from collections import Counter, defaultdict, deque
import networkx as nx
from networkx.utils import groups, not_implemented_for, py_random_state
def _update_label(node, labeling, G):
    """Updates the label of a node using the Prec-Max tie breaking algorithm

    The algorithm is explained in: 'Community Detection via Semi-Synchronous
    Label Propagation Algorithms' Cordasco and Gargano, 2011
    """
    high_labels = _most_frequent_labels(node, labeling, G)
    if len(high_labels) == 1:
        labeling[node] = high_labels.pop()
    elif len(high_labels) > 1:
        if labeling[node] not in high_labels:
            labeling[node] = max(high_labels)