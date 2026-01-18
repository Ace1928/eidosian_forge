import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _compat_shuffle(rng, input):
    """wrapper around rng.shuffle for python 2 compatibility reasons"""
    rng.shuffle(input)