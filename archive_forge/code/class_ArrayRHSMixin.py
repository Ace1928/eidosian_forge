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
class ArrayRHSMixin:

    def __init__(self, lhs, rhs):
        if isinstance(rhs, (tuple, list)) and any(self._rhs_not_none_values(rhs)):
            expressions = []
            for value in rhs:
                if not hasattr(value, 'resolve_expression'):
                    field = lhs.output_field
                    value = Value(field.base_field.get_prep_value(value))
                expressions.append(value)
            rhs = Func(*expressions, function='ARRAY', template='%(function)s[%(expressions)s]')
        super().__init__(lhs, rhs)

    def process_rhs(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        cast_type = self.lhs.output_field.cast_db_type(connection)
        return ('%s::%s' % (rhs, cast_type), rhs_params)

    def _rhs_not_none_values(self, rhs):
        for x in rhs:
            if isinstance(x, (list, tuple)):
                yield from self._rhs_not_none_values(x)
            elif x is not None:
                yield True