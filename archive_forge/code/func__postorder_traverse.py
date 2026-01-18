import collections
import copy
import itertools
import random
import re
import warnings
def _postorder_traverse(root, get_children):
    """Traverse a tree in depth-first post-order (children before parent) (PRIVATE)."""

    def dfs(elem):
        for v in get_children(elem):
            yield from dfs(v)
        yield elem
    yield from dfs(root)