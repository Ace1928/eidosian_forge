import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _apply_forwards_to_bindings(forward, bindings):
    """
    Replace any feature structure that has a forward pointer with
    the target of its forward pointer (to preserve reentrancy).
    """
    for var, value in bindings.items():
        while id(value) in forward:
            value = forward[id(value)]
        bindings[var] = value