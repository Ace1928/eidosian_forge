import math
from django.db.models.expressions import Func, Value
from django.db.models.fields import FloatField, IntegerField
from django.db.models.functions import Cast
from django.db.models.functions.mixins import (
from django.db.models.lookups import Transform
class Radians(NumericOutputFieldMixin, Transform):
    function = 'RADIANS'
    lookup_name = 'radians'

    def as_oracle(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, template='((%%(expressions)s) * %s / 180)' % math.pi, **extra_context)