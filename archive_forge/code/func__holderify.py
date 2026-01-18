from __future__ import annotations
from .. import environment, mparser, mesonlib
from .baseobjects import (
from .exceptions import (
from .decorators import FeatureNew
from .disabler import Disabler, is_disabled
from .helpers import default_resolve_key, flatten, resolve_second_level_holders, stringifyUserArguments
from .operator import MesonOperator
from ._unholder import _unholder
import os, copy, re, pathlib
import typing as T
import textwrap
def _holderify(self, res: T.Union[TYPE_var, InterpreterObject]) -> InterpreterObject:
    if isinstance(res, HoldableTypes):
        cls = self.holder_map.get(type(res), None)
        if cls is not None:
            return cls(res, T.cast('Interpreter', self))
        for typ, cls in self.bound_holder_map.items():
            if isinstance(res, typ):
                return cls(res, T.cast('Interpreter', self))
        raise mesonlib.MesonBugException(f'Object {res} of type {type(res).__name__} is neither in self.holder_map nor self.bound_holder_map.')
    elif isinstance(res, ObjectHolder):
        raise mesonlib.MesonBugException(f'Returned object {res} of type {type(res).__name__} is an object holder.')
    elif isinstance(res, MesonInterpreterObject):
        return res
    raise mesonlib.MesonBugException(f'Unknown returned object {res} of type {type(res).__name__} in the parameters.')