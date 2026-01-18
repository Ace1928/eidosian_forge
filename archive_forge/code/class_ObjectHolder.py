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
class ObjectHolder(InterpreterObject, T.Generic[InterpreterObjectTypeVar]):

    def __init__(self, obj: InterpreterObjectTypeVar, interpreter: 'Interpreter') -> None:
        super().__init__(subproject=interpreter.subproject)
        if not T.TYPE_CHECKING:
            assert isinstance(obj, HoldableTypes), f'This is a bug: Trying to hold object of type `{type(obj).__name__}` that is not in `{HoldableTypes}`'
        self.held_object = obj
        self.interpreter = interpreter
        self.env = self.interpreter.environment

    def display_name(self) -> str:
        return type(self.held_object).__name__

    def op_equals(self, other: TYPE_var) -> bool:
        if type(self.held_object) is not type(other):
            self._throw_comp_exception(other, '==')
        return self.held_object == other

    def op_not_equals(self, other: TYPE_var) -> bool:
        if type(self.held_object) is not type(other):
            self._throw_comp_exception(other, '!=')
        return self.held_object != other

    def __repr__(self) -> str:
        return f'<[{type(self).__name__}] holds [{type(self.held_object).__name__}]: {self.held_object!r}>'