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
def _execute_scalar(self, stmt, type_, parameters=None):
    """Execute a string statement on the current cursor, returning a
        scalar result.

        Used to fire off sequences, default phrases, and "select lastrowid"
        types of statements individually or in the context of a parent INSERT
        or UPDATE statement.

        """
    conn = self.root_connection
    if 'schema_translate_map' in self.execution_options:
        schema_translate_map = self.execution_options.get('schema_translate_map', {})
        rst = self.identifier_preparer._render_schema_translates
        stmt = rst(stmt, schema_translate_map)
    if not parameters:
        if self.dialect.positional:
            parameters = self.dialect.execute_sequence_format()
        else:
            parameters = {}
    conn._cursor_execute(self.cursor, stmt, parameters, context=self)
    row = self.cursor.fetchone()
    if row is not None:
        r = row[0]
    else:
        r = None
    if type_ is not None:
        proc = type_._cached_result_processor(self.dialect, self.cursor.description[0][1])
        if proc:
            return proc(r)
    return r