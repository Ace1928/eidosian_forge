from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def _consume(iterator):
    """Consume the iterator entirely."""
    deque(iterator, maxlen=0)