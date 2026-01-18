from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
def _split_n_from(n, min_size_of_first_part):
    assert n >= min_size_of_first_part
    for p1 in range(min_size_of_first_part, n + 1):
        yield (p1, n - p1)