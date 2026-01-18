import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def _assert_solution_properties(G, aug_edges, avail_dict=None):
    """Checks that aug_edges are consistently formatted"""
    if avail_dict is not None:
        assert all((e in avail_dict for e in aug_edges)), 'when avail is specified aug-edges should be in avail'
    unique_aug = set(map(tuple, map(sorted, aug_edges)))
    unique_aug = list(map(tuple, map(sorted, aug_edges)))
    assert len(aug_edges) == len(unique_aug), 'edges should be unique'
    assert not any((u == v for u, v in unique_aug)), 'should be no self-edges'
    assert not any((G.has_edge(u, v) for u, v in unique_aug)), 'aug edges and G.edges should be disjoint'