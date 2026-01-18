import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _apply_forwards(fstruct, forward, fs_class, visited):
    """
    Replace any feature structure that has a forward pointer with
    the target of its forward pointer (to preserve reentrancy).
    """
    while id(fstruct) in forward:
        fstruct = forward[id(fstruct)]
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError('Expected mapping or sequence')
    for fname, fval in items:
        if isinstance(fval, fs_class):
            while id(fval) in forward:
                fval = forward[id(fval)]
            fstruct[fname] = fval
            _apply_forwards(fval, forward, fs_class, visited)
    return fstruct