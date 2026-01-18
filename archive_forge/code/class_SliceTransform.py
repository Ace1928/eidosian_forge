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
class SliceTransform(Transform):

    def __init__(self, start, end, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end

    def as_sql(self, compiler, connection):
        lhs, params = compiler.compile(self.lhs)
        if not lhs.endswith(']'):
            lhs = '(%s)' % lhs
        return ('%s[%%s:%%s]' % lhs, (*params, self.start, self.end))