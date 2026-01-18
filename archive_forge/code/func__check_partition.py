from itertools import chain
import pytest
import networkx as nx
def _check_partition(G, cut_value, partition, weight):
    assert isinstance(partition, tuple)
    assert len(partition) == 2
    assert isinstance(partition[0], list)
    assert isinstance(partition[1], list)
    assert len(partition[0]) > 0
    assert len(partition[1]) > 0
    assert sum(map(len, partition)) == len(G)
    assert set(chain.from_iterable(partition)) == set(G)
    partition = tuple(map(set, partition))
    w = 0
    for u, v, e in G.edges(data=True):
        if (u in partition[0]) == (v in partition[1]):
            w += e.get(weight, 1)
    assert w == cut_value