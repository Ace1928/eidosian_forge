import collections
import copy
import itertools
import random
import re
import warnings
def _combine_args(first, *rest):
    """Convert ``[targets]`` or ``*targets`` arguments to a single iterable (PRIVATE).

    This helps other functions work like the built-in functions ``max`` and
    ``min``.
    """
    if hasattr(first, '__iter__') and (not isinstance(first, (TreeElement, dict, str, type))):
        if rest:
            raise ValueError('Arguments must be either a single list of targets, or separately specified targets (e.g. foo(t1, t2, t3)), but not both.')
        return first
    return itertools.chain([first], rest)