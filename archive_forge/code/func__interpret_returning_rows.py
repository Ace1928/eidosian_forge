from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
@classmethod
def _interpret_returning_rows(cls, mapper, rows):
    """translate from local inherited table columns to base mapper
        primary key columns.

        Joined inheritance mappers always establish the primary key in terms of
        the base table.   When we UPDATE a sub-table, we can only get
        RETURNING for the sub-table's columns.

        Here, we create a lookup from the local sub table's primary key
        columns to the base table PK columns so that we can get identity
        key values from RETURNING that's against the joined inheritance
        sub-table.

        the complexity here is to support more than one level deep of
        inheritance, where we have to link columns to each other across
        the inheritance hierarchy.

        """
    if mapper.local_table is not mapper.base_mapper.local_table:
        return rows
    local_pk_to_base_pk = {pk: pk for pk in mapper.local_table.primary_key}
    for mp in mapper.iterate_to_root():
        if mp.inherits is None:
            break
        elif mp.local_table is mp.inherits.local_table:
            continue
        t_to_e = dict(mp._table_to_equated[mp.inherits.local_table])
        col_to_col = {sub_pk: super_pk for super_pk, sub_pk in t_to_e[mp]}
        for pk, super_ in local_pk_to_base_pk.items():
            local_pk_to_base_pk[pk] = col_to_col[super_]
    lookup = {local_pk_to_base_pk[lpk]: idx for idx, lpk in enumerate(mapper.local_table.primary_key)}
    primary_key_convert = [lookup[bpk] for bpk in mapper.base_mapper.primary_key]
    return [tuple((row[idx] for idx in primary_key_convert)) for row in rows]