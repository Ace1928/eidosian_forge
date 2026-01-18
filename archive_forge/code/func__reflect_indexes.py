from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
def _reflect_indexes(self, _reflect_info: _ReflectionInfo, table_key: TableKey, table: sa_schema.Table, cols_by_orig_name: Dict[str, sa_schema.Column[Any]], include_columns: Optional[Collection[str]], exclude_columns: Collection[str], reflection_options: Dict[str, Any]) -> None:
    indexes = _reflect_info.indexes.get(table_key, [])
    for index_d in indexes:
        name = index_d['name']
        columns = index_d['column_names']
        expressions = index_d.get('expressions')
        column_sorting = index_d.get('column_sorting', {})
        unique = index_d['unique']
        flavor = index_d.get('type', 'index')
        dialect_options = index_d.get('dialect_options', {})
        duplicates = index_d.get('duplicates_constraint')
        if include_columns and (not set(columns).issubset(include_columns)):
            continue
        if duplicates:
            continue
        idx_element: Any
        idx_elements = []
        for index, c in enumerate(columns):
            if c is None:
                if not expressions:
                    util.warn(f"Skipping {flavor} {name!r} because key {index + 1} reflected as None but no 'expressions' were returned")
                    break
                idx_element = sql.text(expressions[index])
            else:
                try:
                    if c in cols_by_orig_name:
                        idx_element = cols_by_orig_name[c]
                    else:
                        idx_element = table.c[c]
                except KeyError:
                    util.warn(f'{flavor} key {c!r} was not located in columns for table {table.name!r}')
                    continue
                for option in column_sorting.get(c, ()):
                    if option in self._index_sort_exprs:
                        op = self._index_sort_exprs[option]
                        idx_element = op(idx_element)
            idx_elements.append(idx_element)
        else:
            sa_schema.Index(name, *idx_elements, _table=table, unique=unique, **dialect_options)