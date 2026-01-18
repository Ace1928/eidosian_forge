from __future__ import annotations
import collections.abc
import sys
import textwrap
import traceback
from functools import singledispatch
from types import TracebackType
from typing import Any, List, Optional
from ._exceptions import BaseExceptionGroup
def _levenshtein_distance(a, b, max_cost):
    if a == b:
        return 0
    pre = 0
    while a[pre:] and b[pre:] and (a[pre] == b[pre]):
        pre += 1
    a = a[pre:]
    b = b[pre:]
    post = 0
    while a[:post or None] and b[:post or None] and (a[post - 1] == b[post - 1]):
        post -= 1
    a = a[:post or None]
    b = b[:post or None]
    if not a or not b:
        return _MOVE_COST * (len(a) + len(b))
    if len(a) > _MAX_STRING_SIZE or len(b) > _MAX_STRING_SIZE:
        return max_cost + 1
    if len(b) < len(a):
        a, b = (b, a)
    if (len(b) - len(a)) * _MOVE_COST > max_cost:
        return max_cost + 1
    row = list(range(_MOVE_COST, _MOVE_COST * (len(a) + 1), _MOVE_COST))
    result = 0
    for bindex in range(len(b)):
        bchar = b[bindex]
        distance = result = bindex * _MOVE_COST
        minimum = sys.maxsize
        for index in range(len(a)):
            substitute = distance + _substitution_cost(bchar, a[index])
            distance = row[index]
            insert_delete = min(result, distance) + _MOVE_COST
            result = min(insert_delete, substitute)
            row[index] = result
            if result < minimum:
                minimum = result
        if minimum > max_cost:
            return max_cost + 1
    return result