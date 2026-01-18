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
def _call_ambig_func(self, node, data):
    name = node.s.name
    user_func = getattr(self, name, self.__default_ambig__)
    if user_func == self.__default_ambig__ or not hasattr(user_func, 'handles_ambiguity'):
        user_func = partial(self.__default_ambig__, name)
    return user_func(data)