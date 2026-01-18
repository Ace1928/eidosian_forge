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
def _emit_delete_statements(base_mapper, uowtransaction, mapper, table, delete):
    """Emit DELETE statements corresponding to value lists collected
    by _collect_delete_commands()."""
    need_version_id = mapper.version_id_col is not None and mapper.version_id_col in mapper._cols_by_table[table]

    def delete_stmt():
        clauses = BooleanClauseList._construct_raw(operators.and_)
        for col in mapper._pks_by_table[table]:
            clauses._append_inplace(col == sql.bindparam(col.key, type_=col.type))
        if need_version_id:
            clauses._append_inplace(mapper.version_id_col == sql.bindparam(mapper.version_id_col.key, type_=mapper.version_id_col.type))
        return table.delete().where(clauses)
    statement = base_mapper._memo(('delete', table), delete_stmt)
    for connection, recs in groupby(delete, lambda rec: rec[1]):
        del_objects = [params for params, connection in recs]
        execution_options = {'compiled_cache': base_mapper._compiled_cache}
        expected = len(del_objects)
        rows_matched = -1
        only_warn = False
        if need_version_id and (not connection.dialect.supports_sane_multi_rowcount):
            if connection.dialect.supports_sane_rowcount:
                rows_matched = 0
                for params in del_objects:
                    c = connection.execute(statement, params, execution_options=execution_options)
                    rows_matched += c.rowcount
            else:
                util.warn('Dialect %s does not support deleted rowcount - versioning cannot be verified.' % connection.dialect.dialect_description)
                connection.execute(statement, del_objects, execution_options=execution_options)
        else:
            c = connection.execute(statement, del_objects, execution_options=execution_options)
            if not need_version_id:
                only_warn = True
            rows_matched = c.rowcount
        if base_mapper.confirm_deleted_rows and rows_matched > -1 and (expected != rows_matched) and (connection.dialect.supports_sane_multi_rowcount or len(del_objects) == 1):
            if only_warn:
                util.warn("DELETE statement on table '%s' expected to delete %d row(s); %d were matched.  Please set confirm_deleted_rows=False within the mapper configuration to prevent this warning." % (table.description, expected, rows_matched))
            else:
                raise orm_exc.StaleDataError("DELETE statement on table '%s' expected to delete %d row(s); %d were matched.  Please set confirm_deleted_rows=False within the mapper configuration to prevent this warning." % (table.description, expected, rows_matched))