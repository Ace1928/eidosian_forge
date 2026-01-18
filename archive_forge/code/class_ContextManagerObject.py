from __future__ import annotations
from .. import mparser
from .exceptions import InvalidCode, InvalidArguments
from .helpers import flatten, resolve_second_level_holders
from .operator import MesonOperator
from ..mesonlib import HoldableObject, MesonBugException
import textwrap
import typing as T
from abc import ABCMeta
from contextlib import AbstractContextManager
class ContextManagerObject(MesonInterpreterObject, AbstractContextManager):

    def __init__(self, subproject: 'SubProject') -> None:
        super().__init__(subproject=subproject)