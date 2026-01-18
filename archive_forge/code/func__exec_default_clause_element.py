from __future__ import annotations
import functools
import operator
import random
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import characteristics
from . import cursor as _cursor
from . import interfaces
from .base import Connection
from .interfaces import CacheStats
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .reflection import ObjectKind
from .reflection import ObjectScope
from .. import event
from .. import exc
from .. import pool
from .. import util
from ..sql import compiler
from ..sql import dml
from ..sql import expression
from ..sql import type_api
from ..sql._typing import is_tuple_type
from ..sql.base import _NoArg
from ..sql.compiler import DDLCompiler
from ..sql.compiler import InsertmanyvaluesSentinelOpts
from ..sql.compiler import SQLCompiler
from ..sql.elements import quoted_name
from ..util.typing import Final
from ..util.typing import Literal
def _exec_default_clause_element(self, column, default, type_):
    if not default._arg_is_typed:
        default_arg = expression.type_coerce(default.arg, type_)
    else:
        default_arg = default.arg
    compiled = expression.select(default_arg).compile(dialect=self.dialect)
    compiled_params = compiled.construct_params()
    processors = compiled._bind_processors
    if compiled.positional:
        parameters = self.dialect.execute_sequence_format([processors[key](compiled_params[key]) if key in processors else compiled_params[key] for key in compiled.positiontup or ()])
    else:
        parameters = {key: processors[key](compiled_params[key]) if key in processors else compiled_params[key] for key in compiled_params}
    return self._execute_scalar(str(compiled), type_, parameters=parameters)