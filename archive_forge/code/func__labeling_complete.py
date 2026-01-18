from collections import Counter, defaultdict, deque
import networkx as nx
from networkx.utils import groups, not_implemented_for, py_random_state
def _labeling_complete(labeling, G):
    """Determines whether or not LPA is done.

    Label propagation is complete when all nodes have a label that is
    in the set of highest frequency labels amongst its neighbors.

    Nodes with no neighbors are considered complete.
    """
    return all((labeling[v] in _most_frequent_labels(v, labeling, G) for v in G if len(G[v]) > 0))