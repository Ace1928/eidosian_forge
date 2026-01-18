import collections
import copy
import itertools
import random
import re
import warnings
def _level_traverse(root, get_children):
    """Traverse a tree in breadth-first (level) order (PRIVATE)."""
    Q = collections.deque([root])
    while Q:
        v = Q.popleft()
        yield v
        Q.extend(get_children(v))