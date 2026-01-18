import traceback
from collections import deque, namedtuple
from decimal import Decimal
from itertools import chain
from numbers import Number
from pprint import _recursion
from typing import Any, AnyStr, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple  # noqa
from .text import truncate
def _chainlist(it, LIT_LIST_SEP=LIT_LIST_SEP):
    size = len(it)
    for i, v in enumerate(it):
        yield v
        if i < size - 1:
            yield LIT_LIST_SEP