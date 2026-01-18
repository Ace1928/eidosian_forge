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
def _setup_out_parameters(self, result):
    compiled = cast(SQLCompiler, self.compiled)
    out_bindparams = [(param, name) for param, name in compiled.bind_names.items() if param.isoutparam]
    out_parameters = {}
    for bindparam, raw_value in zip([param for param, name in out_bindparams], self.get_out_parameter_values([name for param, name in out_bindparams])):
        type_ = bindparam.type
        impl_type = type_.dialect_impl(self.dialect)
        dbapi_type = impl_type.get_dbapi_type(self.dialect.loaded_dbapi)
        result_processor = impl_type.result_processor(self.dialect, dbapi_type)
        if result_processor is not None:
            raw_value = result_processor(raw_value)
        out_parameters[bindparam.key] = raw_value
    result.out_parameters = out_parameters