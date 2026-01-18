import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def _get_parent_scope_cache(func):
    cache = WeakKeyDictionary()

    def wrapper(parso_cache_node, node, include_flows=False):
        if parso_cache_node is None:
            return func(node, include_flows)
        try:
            for_module = cache[parso_cache_node]
        except KeyError:
            for_module = cache[parso_cache_node] = {}
        try:
            return for_module[node]
        except KeyError:
            result = for_module[node] = func(node, include_flows)
            return result
    return wrapper