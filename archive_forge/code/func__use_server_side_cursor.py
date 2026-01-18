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
def _use_server_side_cursor(self):
    if not self.dialect.supports_server_side_cursors:
        return False
    if self.dialect.server_side_cursors:
        use_server_side = self.execution_options.get('stream_results', True) and (self.compiled and isinstance(self.compiled.statement, expression.Selectable) or ((not self.compiled or isinstance(self.compiled.statement, expression.TextClause)) and self.unicode_statement and SERVER_SIDE_CURSOR_RE.match(self.unicode_statement)))
    else:
        use_server_side = self.execution_options.get('stream_results', False)
    return use_server_side