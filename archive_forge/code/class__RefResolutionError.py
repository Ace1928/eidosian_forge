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
@define(slots=False)
class _RefResolutionError(Exception):
    """
    A ref could not be resolved.
    """
    _DEPRECATION_MESSAGE = 'jsonschema.exceptions.RefResolutionError is deprecated as of version 4.18.0. If you wish to catch potential reference resolution errors, directly catch referencing.exceptions.Unresolvable.'
    _cause: Exception

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self._cause == other._cause

    def __str__(self):
        return str(self._cause)