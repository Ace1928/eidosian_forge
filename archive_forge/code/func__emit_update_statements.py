from __future__ import annotations
from itertools import chain
from itertools import groupby
from itertools import zip_longest
import operator
from . import attributes
from . import exc as orm_exc
from . import loading
from . import sync
from .base import state_str
from .. import exc as sa_exc
from .. import future
from .. import sql
from .. import util
from ..engine import cursor as _cursor
from ..sql import operators
from ..sql.elements import BooleanClauseList
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
def _emit_update_statements(base_mapper, uowtransaction, mapper, table, update, *, bookkeeping=True, use_orm_update_stmt=None, enable_check_rowcount=True):
    """Emit UPDATE statements corresponding to value lists collected
    by _collect_update_commands()."""
    needs_version_id = mapper.version_id_col is not None and mapper.version_id_col in mapper._cols_by_table[table]
    execution_options = {'compiled_cache': base_mapper._compiled_cache}

    def update_stmt(existing_stmt=None):
        clauses = BooleanClauseList._construct_raw(operators.and_)
        for col in mapper._pks_by_table[table]:
            clauses._append_inplace(col == sql.bindparam(col._label, type_=col.type))
        if needs_version_id:
            clauses._append_inplace(mapper.version_id_col == sql.bindparam(mapper.version_id_col._label, type_=mapper.version_id_col.type))
        if existing_stmt is not None:
            stmt = existing_stmt.where(clauses)
        else:
            stmt = table.update().where(clauses)
        return stmt
    if use_orm_update_stmt is not None:
        cached_stmt = update_stmt(use_orm_update_stmt)
    else:
        cached_stmt = base_mapper._memo(('update', table), update_stmt)
    for (connection, paramkeys, hasvalue, has_all_defaults, has_all_pks), records in groupby(update, lambda rec: (rec[4], set(rec[2]), bool(rec[5]), rec[6], rec[7])):
        rows = 0
        records = list(records)
        statement = cached_stmt
        if use_orm_update_stmt is not None:
            statement = statement._annotate({'_emit_update_table': table, '_emit_update_mapper': mapper})
        return_defaults = False
        if not has_all_pks:
            statement = statement.return_defaults(*mapper._pks_by_table[table])
            return_defaults = True
        if bookkeeping and (not has_all_defaults) and (mapper.base_mapper.eager_defaults is True) and table.implicit_returning and connection.dialect.update_returning:
            statement = statement.return_defaults(*mapper._server_onupdate_default_cols[table])
            return_defaults = True
        if mapper._version_id_has_server_side_value:
            statement = statement.return_defaults(mapper.version_id_col)
            return_defaults = True
        assert_singlerow = connection.dialect.supports_sane_rowcount
        assert_multirow = assert_singlerow and connection.dialect.supports_sane_multi_rowcount
        allow_executemany = not return_defaults and (not needs_version_id)
        if hasvalue:
            for state, state_dict, params, mapper, connection, value_params, has_all_defaults, has_all_pks in records:
                c = connection.execute(statement.values(value_params), params, execution_options=execution_options)
                if bookkeeping:
                    _postfetch(mapper, uowtransaction, table, state, state_dict, c, c.context.compiled_parameters[0], value_params, True, c.returned_defaults)
                rows += c.rowcount
                check_rowcount = enable_check_rowcount and assert_singlerow
        elif not allow_executemany:
            check_rowcount = enable_check_rowcount and assert_singlerow
            for state, state_dict, params, mapper, connection, value_params, has_all_defaults, has_all_pks in records:
                c = connection.execute(statement, params, execution_options=execution_options)
                if bookkeeping:
                    _postfetch(mapper, uowtransaction, table, state, state_dict, c, c.context.compiled_parameters[0], value_params, True, c.returned_defaults)
                rows += c.rowcount
        else:
            multiparams = [rec[2] for rec in records]
            check_rowcount = enable_check_rowcount and (assert_multirow or (assert_singlerow and len(multiparams) == 1))
            c = connection.execute(statement, multiparams, execution_options=execution_options)
            rows += c.rowcount
            for state, state_dict, params, mapper, connection, value_params, has_all_defaults, has_all_pks in records:
                if bookkeeping:
                    _postfetch(mapper, uowtransaction, table, state, state_dict, c, c.context.compiled_parameters[0], value_params, True, c.returned_defaults if not c.context.executemany else None)
        if check_rowcount:
            if rows != len(records):
                raise orm_exc.StaleDataError("UPDATE statement on table '%s' expected to update %d row(s); %d were matched." % (table.description, len(records), rows))
        elif needs_version_id:
            util.warn('Dialect %s does not support updated rowcount - versioning cannot be verified.' % c.dialect.dialect_description)