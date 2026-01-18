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
class StrCompileDialect(DefaultDialect):
    statement_compiler = compiler.StrSQLCompiler
    ddl_compiler = compiler.DDLCompiler
    type_compiler_cls = compiler.StrSQLTypeCompiler
    preparer = compiler.IdentifierPreparer
    insert_returning = True
    update_returning = True
    delete_returning = True
    supports_statement_cache = True
    supports_identity_columns = True
    supports_sequences = True
    sequences_optional = True
    preexecute_autoincrement_sequences = False
    supports_native_boolean = True
    supports_multivalues_insert = True
    supports_simple_order_by_label = True