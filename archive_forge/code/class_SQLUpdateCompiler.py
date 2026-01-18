from django.core.exceptions import FieldError, FullResultSet
from django.db.models.expressions import Col
from django.db.models.sql import compiler
class SQLUpdateCompiler(compiler.SQLUpdateCompiler, SQLCompiler):

    def as_sql(self):
        update_query, update_params = super().as_sql()
        if self.query.order_by:
            order_by_sql = []
            order_by_params = []
            db_table = self.query.get_meta().db_table
            try:
                for resolved, (sql, params, _) in self.get_order_by():
                    if isinstance(resolved.expression, Col) and resolved.expression.alias != db_table:
                        raise FieldError
                    order_by_sql.append(sql)
                    order_by_params.extend(params)
                update_query += ' ORDER BY ' + ', '.join(order_by_sql)
                update_params += tuple(order_by_params)
            except FieldError:
                pass
        return (update_query, update_params)