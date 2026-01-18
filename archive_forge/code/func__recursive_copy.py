import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def _recursive_copy(self, node):
    copied_node = {}
    for key, value in node.items():
        if isinstance(value, NodeList):
            copied_node[key] = copy.copy(value)
        elif isinstance(value, dict):
            copied_node[key] = self._recursive_copy(value)
        else:
            copied_node[key] = value
    return copied_node