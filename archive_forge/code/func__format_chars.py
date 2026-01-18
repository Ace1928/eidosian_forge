import traceback
from collections import deque, namedtuple
from decimal import Decimal
from itertools import chain
from numbers import Number
from pprint import _recursion
from typing import Any, AnyStr, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple  # noqa
from .text import truncate
def _format_chars(val, maxlen):
    if isinstance(val, bytes):
        return _format_binary_bytes(val, maxlen)
    else:
        return "'{}'".format(truncate(val, maxlen).replace("'", "\\'"))