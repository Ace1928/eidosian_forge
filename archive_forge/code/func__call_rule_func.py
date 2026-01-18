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
def _call_rule_func(self, node, data):
    name = node.rule.alias or node.rule.options.template_source or node.rule.origin.name
    user_func = getattr(self, name, self.__default__)
    if user_func == self.__default__ or hasattr(user_func, 'handles_ambiguity'):
        user_func = partial(self.__default__, name)
    if not self.resolve_ambiguity:
        wrapper = partial(AmbiguousIntermediateExpander, self.tree_class)
        user_func = wrapper(user_func)
    return user_func(data)