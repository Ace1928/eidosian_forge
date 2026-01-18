from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree
def __default__(self, name, data):
    """Default operation on tree (for override).

        Returns a tree with name with data as children.
        """
    return self.tree_class(name, data)