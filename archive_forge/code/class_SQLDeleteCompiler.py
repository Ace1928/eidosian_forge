from django.core.exceptions import FieldError, FullResultSet
from django.db.models.expressions import Col
from django.db.models.sql import compiler
class SQLDeleteCompiler(compiler.SQLDeleteCompiler, SQLCompiler):

    def as_sql(self):
        where, having, qualify = self.query.where.split_having_qualify(must_group_by=self.query.group_by is not None)
        if self.single_alias or having or qualify:
            return super().as_sql()
        result = ['DELETE %s FROM' % self.quote_name_unless_alias(self.query.get_initial_alias())]
        from_sql, params = self.get_from_clause()
        result.extend(from_sql)
        try:
            where_sql, where_params = self.compile(where)
        except FullResultSet:
            pass
        else:
            result.append('WHERE %s' % where_sql)
            params.extend(where_params)
        return (' '.join(result), tuple(params))