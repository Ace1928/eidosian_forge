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
def _check_cycle(self, node):
    if self._on_cycle_retreat:
        if id(node) == id(self._cycle_node) or id(node) in self._successful_visits:
            self._cycle_node = None
            self._on_cycle_retreat = False
        else:
            return Discard