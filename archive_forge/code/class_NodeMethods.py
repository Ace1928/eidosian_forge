import ast
import collections
import io
import sys
import token
import tokenize
from abc import ABCMeta
from ast import Module, expr, AST
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast, Any, TYPE_CHECKING
from six import iteritems
class NodeMethods(object):
    """
  Helper to get `visit_{node_type}` methods given a node's class and cache the results.
  """

    def __init__(self):
        self._cache = {}

    def get(self, obj, cls):
        """
    Using the lowercase name of the class as node_type, returns `obj.visit_{node_type}`,
    or `obj.visit_default` if the type-specific method is not found.
    """
        method = self._cache.get(cls)
        if not method:
            name = 'visit_' + cls.__name__.lower()
            method = getattr(obj, name, obj.visit_default)
            self._cache[cls] = method
        return method