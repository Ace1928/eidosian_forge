from __future__ import annotations
import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Sequence, overload
from typing_extensions import Self  # Python 3.11+
from optree import _C
from optree.typing import (
from optree.utils import safe_zip, total_order_sorted, unzip2
def _structseq_flatten(seq: structseq[T]) -> tuple[tuple[T, ...], type[structseq[T]]]:
    return (seq, type(seq))