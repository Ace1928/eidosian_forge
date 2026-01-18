import collections
import copy
import itertools
import random
import re
import warnings
def _attribute_matcher(kwargs):
    """Match a node by specified attribute values (PRIVATE).

    ``terminal`` is a special case: True restricts the search to external (leaf)
    nodes, False restricts to internal nodes, and None allows all tree elements
    to be searched, including phyloXML annotations.

    Otherwise, for a tree element to match the specification (i.e. for the
    function produced by ``_attribute_matcher`` to return True when given a tree
    element), it must have each of the attributes specified by the keys and
    match each of the corresponding values -- think 'and', not 'or', for
    multiple keys.
    """

    def match(node):
        if 'terminal' in kwargs:
            kwa_copy = kwargs.copy()
            pattern = kwa_copy.pop('terminal')
            if pattern is not None and (not hasattr(node, 'is_terminal') or node.is_terminal() != pattern):
                return False
        else:
            kwa_copy = kwargs
        for key, pattern in kwa_copy.items():
            if not hasattr(node, key):
                return False
            target = getattr(node, key)
            if isinstance(pattern, str):
                return isinstance(target, str) and re.match(pattern + '$', target)
            if isinstance(pattern, bool):
                return pattern == bool(target)
            if isinstance(pattern, int):
                return pattern == target
            if pattern is None:
                return target is None
            raise TypeError(f'invalid query type: {type(pattern)}')
        return True
    return match