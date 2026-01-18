from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
def _check_attach(self, evt: bool=False) -> None:
    col_objs = [c for c in self._pending_colargs if isinstance(c, Column)]
    cols_w_table = [c for c in col_objs if isinstance(c.table, Table)]
    cols_wo_table = set(col_objs).difference(cols_w_table)
    if cols_wo_table:
        assert not evt, 'Should not reach here on event call'
        has_string_cols = {c for c in self._pending_colargs if c is not None}.difference(col_objs)
        if not has_string_cols:

            def _col_attached(column: Column[Any], table: Table) -> None:
                if isinstance(table, Table):
                    cols_wo_table.discard(column)
                    if not cols_wo_table:
                        self._check_attach(evt=True)
            self._cols_wo_table = cols_wo_table
            for col in cols_wo_table:
                col._on_table_attach(_col_attached)
            return
    columns = cols_w_table
    tables = {c.table for c in columns}
    if len(tables) == 1:
        self._set_parent_with_dispatch(tables.pop())
    elif len(tables) > 1 and (not self._allow_multiple_tables):
        table = columns[0].table
        others = [c for c in columns[1:] if c.table is not table]
        if others:
            other_str = ', '.join(("'%s'" % c for c in others))
            raise exc.ArgumentError(f"Column(s) {other_str} are not part of table '{table.description}'.")