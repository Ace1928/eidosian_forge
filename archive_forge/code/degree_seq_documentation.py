import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
Returns True if and only if an arbitrary remaining node can
        potentially be joined with some other remaining node.

        