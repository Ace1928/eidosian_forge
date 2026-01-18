import collections.abc
import os
import pprint
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Sequence
from unicodedata import normalize
from _pytest import outcomes
import _pytest._code
from _pytest._io.pprint import PrettyPrinter
from _pytest._io.saferepr import saferepr
from _pytest._io.saferepr import saferepr_unlimited
from _pytest.config import Config
def _compare_eq_sequence(left: Sequence[Any], right: Sequence[Any], highlighter: _HighlightFunc, verbose: int=0) -> List[str]:
    comparing_bytes = isinstance(left, bytes) and isinstance(right, bytes)
    explanation: List[str] = []
    len_left = len(left)
    len_right = len(right)
    for i in range(min(len_left, len_right)):
        if left[i] != right[i]:
            if comparing_bytes:
                left_value = left[i:i + 1]
                right_value = right[i:i + 1]
            else:
                left_value = left[i]
                right_value = right[i]
            explanation.append(f'At index {i} diff: {highlighter(repr(left_value))} != {highlighter(repr(right_value))}')
            break
    if comparing_bytes:
        return explanation
    len_diff = len_left - len_right
    if len_diff:
        if len_diff > 0:
            dir_with_more = 'Left'
            extra = saferepr(left[len_right])
        else:
            len_diff = 0 - len_diff
            dir_with_more = 'Right'
            extra = saferepr(right[len_left])
        if len_diff == 1:
            explanation += [f'{dir_with_more} contains one more item: {highlighter(extra)}']
        else:
            explanation += ['%s contains %d more items, first extra item: %s' % (dir_with_more, len_diff, highlighter(extra))]
    return explanation