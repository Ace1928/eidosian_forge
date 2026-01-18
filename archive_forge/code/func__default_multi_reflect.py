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
def _default_multi_reflect(self, single_tbl_method, connection, kind, schema, filter_names, scope, **kw):
    names_fns = []
    temp_names_fns = []
    if ObjectKind.TABLE in kind:
        names_fns.append(self.get_table_names)
        temp_names_fns.append(self.get_temp_table_names)
    if ObjectKind.VIEW in kind:
        names_fns.append(self.get_view_names)
        temp_names_fns.append(self.get_temp_view_names)
    if ObjectKind.MATERIALIZED_VIEW in kind:
        names_fns.append(self.get_materialized_view_names)
    unreflectable = kw.pop('unreflectable', {})
    if filter_names and scope is ObjectScope.ANY and (kind is ObjectKind.ANY):
        names = filter_names
    else:
        names = []
        name_kw = {'schema': schema, **kw}
        fns = []
        if ObjectScope.DEFAULT in scope:
            fns.extend(names_fns)
        if ObjectScope.TEMPORARY in scope:
            fns.extend(temp_names_fns)
        for fn in fns:
            try:
                names.extend(fn(connection, **name_kw))
            except NotImplementedError:
                pass
    if filter_names:
        filter_names = set(filter_names)
    for table in names:
        if not filter_names or table in filter_names:
            key = (schema, table)
            try:
                yield (key, single_tbl_method(connection, table, schema=schema, **kw))
            except exc.UnreflectableTableError as err:
                if key not in unreflectable:
                    unreflectable[key] = err
            except exc.NoSuchTableError:
                pass