import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _try_getitem(d):
    try:
        return d[weight]
    except TypeError:
        return d