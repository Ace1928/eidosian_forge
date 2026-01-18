from django.db import NotSupportedError
from django.db.models.expressions import Func, Value
from django.db.models.fields import CharField, IntegerField, TextField
from django.db.models.functions import Cast, Coalesce
from django.db.models.lookups import Transform
class MySQLSHA2Mixin:

    def as_mysql(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, template='SHA2(%%(expressions)s, %s)' % self.function[3:], **extra_context)