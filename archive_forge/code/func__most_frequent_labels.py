from collections import Counter, defaultdict, deque
import networkx as nx
from networkx.utils import groups, not_implemented_for, py_random_state
def _most_frequent_labels(node, labeling, G):
    """Returns a set of all labels with maximum frequency in `labeling`.

    Input `labeling` should be a dict keyed by node to labels.
    """
    if not G[node]:
        return {labeling[node]}
    freqs = Counter((labeling[q] for q in G[node]))
    max_freq = max(freqs.values())
    return {label for label, freq in freqs.items() if freq == max_freq}