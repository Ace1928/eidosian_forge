import json
import warnings
from django import forms
from django.core import checks, exceptions
from django.db import NotSupportedError, connections, router
from django.db.models import expressions, lookups
from django.db.models.constants import LOOKUP_SEP
from django.db.models.fields import TextField
from django.db.models.lookups import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import CheckFieldDefaultMixin
class KeyTransformExact(JSONExact):

    def process_rhs(self, compiler, connection):
        if isinstance(self.rhs, KeyTransform):
            return super(lookups.Exact, self).process_rhs(compiler, connection)
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if connection.vendor == 'oracle':
            func = []
            sql = "%s(JSON_OBJECT('value' VALUE %%s FORMAT JSON), '$.value')"
            for value in rhs_params:
                value = json.loads(value)
                if isinstance(value, (list, dict)):
                    func.append(sql % 'JSON_QUERY')
                else:
                    func.append(sql % 'JSON_VALUE')
            rhs %= tuple(func)
        elif connection.vendor == 'sqlite':
            func = []
            for value in rhs_params:
                if value in connection.ops.jsonfield_datatype_values:
                    func.append('%s')
                else:
                    func.append("JSON_EXTRACT(%s, '$')")
            rhs %= tuple(func)
        return (rhs, rhs_params)

    def as_oracle(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if rhs_params == ['null']:
            has_key_expr = HasKeyOrArrayIndex(self.lhs.lhs, self.lhs.key_name)
            has_key_sql, has_key_params = has_key_expr.as_oracle(compiler, connection)
            is_null_expr = self.lhs.get_lookup('isnull')(self.lhs, True)
            is_null_sql, is_null_params = is_null_expr.as_sql(compiler, connection)
            return ('%s AND %s' % (has_key_sql, is_null_sql), tuple(has_key_params) + tuple(is_null_params))
        return super().as_sql(compiler, connection)