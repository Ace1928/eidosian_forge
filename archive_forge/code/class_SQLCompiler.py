from django.core.exceptions import FieldError, FullResultSet
from django.db.models.expressions import Col
from django.db.models.sql import compiler
class SQLCompiler(compiler.SQLCompiler):

    def as_subquery_condition(self, alias, columns, compiler):
        qn = compiler.quote_name_unless_alias
        qn2 = self.connection.ops.quote_name
        sql, params = self.as_sql()
        return ('(%s) IN (%s)' % (', '.join(('%s.%s' % (qn(alias), qn2(column)) for column in columns)), sql), params)