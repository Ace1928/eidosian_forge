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
def _unholder_args(self, args: T.List[InterpreterObject], kwargs: T.Dict[str, InterpreterObject]) -> T.Tuple[T.List[TYPE_var], TYPE_kwargs]:
    return ([_unholder(x) for x in args], {k: _unholder(v) for k, v in kwargs.items()})