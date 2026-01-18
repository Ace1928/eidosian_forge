from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def _check_antichains(self, solution, result):
    sol = [frozenset(a) for a in solution]
    res = [frozenset(a) for a in result]
    assert set(sol) == set(res)