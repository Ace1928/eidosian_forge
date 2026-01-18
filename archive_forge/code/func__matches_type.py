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
def _matches_type(self):
    try:
        expected = self.schema['type']
    except (KeyError, TypeError):
        return False
    if isinstance(expected, str):
        return self._type_checker.is_type(self.instance, expected)
    return any((self._type_checker.is_type(self.instance, expected_type) for expected_type in expected))