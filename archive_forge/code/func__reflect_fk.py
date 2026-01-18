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
def _reflect_fk(self, _reflect_info: _ReflectionInfo, table_key: TableKey, table: sa_schema.Table, cols_by_orig_name: Dict[str, sa_schema.Column[Any]], include_columns: Optional[Collection[str]], exclude_columns: Collection[str], resolve_fks: bool, _extend_on: Optional[Set[sa_schema.Table]], reflection_options: Dict[str, Any]) -> None:
    fkeys = _reflect_info.foreign_keys.get(table_key, [])
    for fkey_d in fkeys:
        conname = fkey_d['name']
        constrained_columns = [cols_by_orig_name[c].key if c in cols_by_orig_name else c for c in fkey_d['constrained_columns']]
        if exclude_columns and set(constrained_columns).intersection(exclude_columns) or (include_columns and set(constrained_columns).difference(include_columns)):
            continue
        referred_schema = fkey_d['referred_schema']
        referred_table = fkey_d['referred_table']
        referred_columns = fkey_d['referred_columns']
        refspec = []
        if referred_schema is not None:
            if resolve_fks:
                sa_schema.Table(referred_table, table.metadata, schema=referred_schema, autoload_with=self.bind, _extend_on=_extend_on, _reflect_info=_reflect_info, **reflection_options)
            for column in referred_columns:
                refspec.append('.'.join([referred_schema, referred_table, column]))
        else:
            if resolve_fks:
                sa_schema.Table(referred_table, table.metadata, autoload_with=self.bind, schema=sa_schema.BLANK_SCHEMA, _extend_on=_extend_on, _reflect_info=_reflect_info, **reflection_options)
            for column in referred_columns:
                refspec.append('.'.join([referred_table, column]))
        if 'options' in fkey_d:
            options = fkey_d['options']
        else:
            options = {}
        try:
            table.append_constraint(sa_schema.ForeignKeyConstraint(constrained_columns, refspec, conname, link_to_name=True, comment=fkey_d.get('comment'), **options))
        except exc.ConstraintColumnNotFoundError:
            util.warn(f'On reflected table {table.name}, skipping reflection of foreign key constraint {conname}; one or more subject columns within name(s) {', '.join(constrained_columns)} are not present in the table')