from django.db import NotSupportedError
from django.db.models.expressions import Func, Value
from django.db.models.fields import CharField, IntegerField, TextField
from django.db.models.functions import Cast, Coalesce
from django.db.models.lookups import Transform
class PostgreSQLSHAMixin:

    def as_postgresql(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, template="ENCODE(DIGEST(%(expressions)s, '%(function)s'), 'hex')", function=self.function.lower(), **extra_context)