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
class UndefinedTypeCheck(Exception):
    """
    A type checker was asked to check a type it did not have registered.
    """

    def __init__(self, type):
        self.type = type

    def __str__(self):
        return f'Type {self.type!r} is unknown to this type checker'