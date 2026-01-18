from django.db import NotSupportedError
from django.db.models.expressions import Func, Value
from django.db.models.fields import CharField, IntegerField, TextField
from django.db.models.functions import Cast, Coalesce
from django.db.models.lookups import Transform
class Chr(Transform):
    function = 'CHR'
    lookup_name = 'chr'
    output_field = CharField()

    def as_mysql(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, function='CHAR', template='%(function)s(%(expressions)s USING utf16)', **extra_context)

    def as_oracle(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, template='%(function)s(%(expressions)s USING NCHAR_CS)', **extra_context)

    def as_sqlite(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, function='CHAR', **extra_context)