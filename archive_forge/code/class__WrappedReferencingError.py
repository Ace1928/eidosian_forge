from __future__ import annotations
from collections import defaultdict, deque
from pprint import pformat
from textwrap import dedent, indent
from typing import TYPE_CHECKING, ClassVar
import heapq
import itertools
import warnings
from attrs import define
from referencing.exceptions import Unresolvable as _Unresolvable
from jsonschema import _utils
class _WrappedReferencingError(_RefResolutionError, _Unresolvable):

    def __init__(self, cause: _Unresolvable):
        object.__setattr__(self, '_wrapped', cause)

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self._wrapped == other._wrapped
        elif other.__class__ is self._wrapped.__class__:
            return self._wrapped == other
        return NotImplemented

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)

    def __hash__(self):
        return hash(self._wrapped)

    def __repr__(self):
        return f'<WrappedReferencingError {self._wrapped!r}>'

    def __str__(self):
        return f'{self._wrapped.__class__.__name__}: {self._wrapped}'