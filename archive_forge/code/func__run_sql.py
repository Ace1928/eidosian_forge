from django.db import router
from .base import Operation
def _run_sql(self, schema_editor, sqls):
    if isinstance(sqls, (list, tuple)):
        for sql in sqls:
            params = None
            if isinstance(sql, (list, tuple)):
                elements = len(sql)
                if elements == 2:
                    sql, params = sql
                else:
                    raise ValueError('Expected a 2-tuple but got %d' % elements)
            schema_editor.execute(sql, params=params)
    elif sqls != RunSQL.noop:
        statements = schema_editor.connection.ops.prepare_sql_script(sqls)
        for statement in statements:
            schema_editor.execute(statement, params=None)