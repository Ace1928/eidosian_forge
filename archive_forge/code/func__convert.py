from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _convert(obj, attributes, format_references):
    definition = {}
    for key in attributes:
        val = getattr(obj, key, False)
        if not val:
            continue
        if isinstance(val, string_types):
            definition[key] = val
        else:
            try:
                definition[key] = [rep(v, format_references) for v in iter(val)]
            except TypeError:
                definition[key] = rep(val, format_references)
    return definition