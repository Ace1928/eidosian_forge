from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
Recursively creates a tree from the given sequence of nested
        tuples.

        This function employs the :func:`~networkx.tree.join` function
        to recursively join subtrees into a larger tree.

        