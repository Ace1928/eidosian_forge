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
def _process_execute_defaults(self):
    compiled = cast(SQLCompiler, self.compiled)
    key_getter = compiled._within_exec_param_key_getter
    sentinel_counter = 0
    if compiled.insert_prefetch:
        prefetch_recs = [(c, key_getter(c), c._default_description_tuple, self.get_insert_default) for c in compiled.insert_prefetch]
    elif compiled.update_prefetch:
        prefetch_recs = [(c, key_getter(c), c._onupdate_description_tuple, self.get_update_default) for c in compiled.update_prefetch]
    else:
        prefetch_recs = []
    for param in self.compiled_parameters:
        self.current_parameters = param
        for c, param_key, (arg, is_scalar, is_callable, is_sentinel), fallback in prefetch_recs:
            if is_sentinel:
                param[param_key] = sentinel_counter
                sentinel_counter += 1
            elif is_scalar:
                param[param_key] = arg
            elif is_callable:
                self.current_column = c
                param[param_key] = arg(self)
            else:
                val = fallback(c)
                if val is not None:
                    param[param_key] = val
    del self.current_parameters