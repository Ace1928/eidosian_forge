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
def _compare_eq_dict(left: Mapping[Any, Any], right: Mapping[Any, Any], highlighter: _HighlightFunc, verbose: int=0) -> List[str]:
    explanation: List[str] = []
    set_left = set(left)
    set_right = set(right)
    common = set_left.intersection(set_right)
    same = {k: left[k] for k in common if left[k] == right[k]}
    if same and verbose < 2:
        explanation += ['Omitting %s identical items, use -vv to show' % len(same)]
    elif same:
        explanation += ['Common items:']
        explanation += highlighter(pprint.pformat(same)).splitlines()
    diff = {k for k in common if left[k] != right[k]}
    if diff:
        explanation += ['Differing items:']
        for k in diff:
            explanation += [highlighter(saferepr({k: left[k]})) + ' != ' + highlighter(saferepr({k: right[k]}))]
    extra_left = set_left - set_right
    len_extra_left = len(extra_left)
    if len_extra_left:
        explanation.append('Left contains %d more item%s:' % (len_extra_left, '' if len_extra_left == 1 else 's'))
        explanation.extend(highlighter(pprint.pformat({k: left[k] for k in extra_left})).splitlines())
    extra_right = set_right - set_left
    len_extra_right = len(extra_right)
    if len_extra_right:
        explanation.append('Right contains %d more item%s:' % (len_extra_right, '' if len_extra_right == 1 else 's'))
        explanation.extend(highlighter(pprint.pformat({k: right[k] for k in extra_right})).splitlines())
    return explanation