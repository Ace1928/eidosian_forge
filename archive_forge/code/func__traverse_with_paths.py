import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
def _traverse_with_paths(self):
    """Yield each item with its path in the tree."""
    children = getattr(self, 'children', None)
    children_type = type(children).__name__
    children_string = children_type + self._id_str(children)
    if isinstance(children, Component):
        yield ('[*] ' + children_string, children)
        for p, t in children._traverse_with_paths():
            yield ('\n'.join(['[*] ' + children_string, p]), t)
    elif isinstance(children, (tuple, MutableSequence)):
        for idx, i in enumerate(children):
            list_path = f'[{idx:d}] {type(i).__name__:s}{self._id_str(i)}'
            yield (list_path, i)
            if isinstance(i, Component):
                for p, t in i._traverse_with_paths():
                    yield ('\n'.join([list_path, p]), t)