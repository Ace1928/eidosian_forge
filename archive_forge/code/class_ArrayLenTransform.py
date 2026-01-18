import json
from django.contrib.postgres import lookups
from django.contrib.postgres.forms import SimpleArrayField
from django.contrib.postgres.validators import ArrayMaxLengthValidator
from django.core import checks, exceptions
from django.db.models import Field, Func, IntegerField, Transform, Value
from django.db.models.fields.mixins import CheckFieldDefaultMixin
from django.db.models.lookups import Exact, In
from django.utils.translation import gettext_lazy as _
from ..utils import prefix_validation_error
from .utils import AttributeSetter
@ArrayField.register_lookup
class ArrayLenTransform(Transform):
    lookup_name = 'len'
    output_field = IntegerField()

    def as_sql(self, compiler, connection):
        lhs, params = compiler.compile(self.lhs)
        return ('CASE WHEN %(lhs)s IS NULL THEN NULL ELSE coalesce(array_length(%(lhs)s, 1), 0) END' % {'lhs': lhs}, params * 2)