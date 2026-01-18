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
def _bulk_insert(mapper: Mapper[_O], mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]], session_transaction: SessionTransaction, isstates: bool, return_defaults: bool, render_nulls: bool, use_orm_insert_stmt: Optional[dml.Insert]=None, execution_options: Optional[OrmExecuteOptionsParameter]=None) -> Optional[cursor.CursorResult[Any]]:
    base_mapper = mapper.base_mapper
    if session_transaction.session.connection_callable:
        raise NotImplementedError('connection_callable / per-instance sharding not supported in bulk_insert()')
    if isstates:
        if return_defaults:
            states = [(state, state.dict) for state in mappings]
            mappings = [dict_ for state, dict_ in states]
        else:
            mappings = [state.dict for state in mappings]
    else:
        mappings = [dict(m) for m in mappings]
        _expand_composites(mapper, mappings)
    connection = session_transaction.connection(base_mapper)
    return_result: Optional[cursor.CursorResult[Any]] = None
    mappers_to_run = [(table, mp) for table, mp in base_mapper._sorted_tables.items() if table in mapper._pks_by_table]
    if return_defaults:
        bookkeeping = True
    elif len(mappers_to_run) > 1:
        bookkeeping = True
    else:
        bookkeeping = False
    for table, super_mapper in mappers_to_run:
        extra_bp_names = [b.key for b in use_orm_insert_stmt._get_embedded_bindparams() if b.key in mappings[0]] if use_orm_insert_stmt is not None else ()
        records = ((None, state_dict, params, mapper, connection, value_params, has_all_pks, has_all_defaults) for state, state_dict, params, mp, conn, value_params, has_all_pks, has_all_defaults in persistence._collect_insert_commands(table, ((None, mapping, mapper, connection) for mapping in mappings), bulk=True, return_defaults=bookkeeping, render_nulls=render_nulls, include_bulk_keys=extra_bp_names))
        result = persistence._emit_insert_statements(base_mapper, None, super_mapper, table, records, bookkeeping=bookkeeping, use_orm_insert_stmt=use_orm_insert_stmt, execution_options=execution_options)
        if use_orm_insert_stmt is not None:
            if not use_orm_insert_stmt._returning or return_result is None:
                return_result = result
            elif result.returns_rows:
                assert bookkeeping
                return_result = return_result.splice_horizontally(result)
    if return_defaults and isstates:
        identity_cls = mapper._identity_class
        identity_props = [p.key for p in mapper._identity_key_props]
        for state, dict_ in states:
            state.key = (identity_cls, tuple([dict_[key] for key in identity_props]))
    if use_orm_insert_stmt is not None:
        assert return_result is not None
        return return_result