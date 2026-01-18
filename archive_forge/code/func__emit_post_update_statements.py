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
def _emit_post_update_statements(base_mapper, uowtransaction, mapper, table, update):
    """Emit UPDATE statements corresponding to value lists collected
    by _collect_post_update_commands()."""
    execution_options = {'compiled_cache': base_mapper._compiled_cache}
    needs_version_id = mapper.version_id_col is not None and mapper.version_id_col in mapper._cols_by_table[table]

    def update_stmt():
        clauses = BooleanClauseList._construct_raw(operators.and_)
        for col in mapper._pks_by_table[table]:
            clauses._append_inplace(col == sql.bindparam(col._label, type_=col.type))
        if needs_version_id:
            clauses._append_inplace(mapper.version_id_col == sql.bindparam(mapper.version_id_col._label, type_=mapper.version_id_col.type))
        stmt = table.update().where(clauses)
        return stmt
    statement = base_mapper._memo(('post_update', table), update_stmt)
    if mapper._version_id_has_server_side_value:
        statement = statement.return_defaults(mapper.version_id_col)
    for key, records in groupby(update, lambda rec: (rec[3], set(rec[4]))):
        rows = 0
        records = list(records)
        connection = key[0]
        assert_singlerow = connection.dialect.supports_sane_rowcount
        assert_multirow = assert_singlerow and connection.dialect.supports_sane_multi_rowcount
        allow_executemany = not needs_version_id or assert_multirow
        if not allow_executemany:
            check_rowcount = assert_singlerow
            for state, state_dict, mapper_rec, connection, params in records:
                c = connection.execute(statement, params, execution_options=execution_options)
                _postfetch_post_update(mapper_rec, uowtransaction, table, state, state_dict, c, c.context.compiled_parameters[0])
                rows += c.rowcount
        else:
            multiparams = [params for state, state_dict, mapper_rec, conn, params in records]
            check_rowcount = assert_multirow or (assert_singlerow and len(multiparams) == 1)
            c = connection.execute(statement, multiparams, execution_options=execution_options)
            rows += c.rowcount
            for state, state_dict, mapper_rec, connection, params in records:
                _postfetch_post_update(mapper_rec, uowtransaction, table, state, state_dict, c, c.context.compiled_parameters[0])
        if check_rowcount:
            if rows != len(records):
                raise orm_exc.StaleDataError("UPDATE statement on table '%s' expected to update %d row(s); %d were matched." % (table.description, len(records), rows))
        elif needs_version_id:
            util.warn('Dialect %s does not support updated rowcount - versioning cannot be verified.' % c.dialect.dialect_description)