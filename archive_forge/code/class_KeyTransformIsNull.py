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
class KeyTransformIsNull(lookups.IsNull):

    def as_oracle(self, compiler, connection):
        sql, params = HasKeyOrArrayIndex(self.lhs.lhs, self.lhs.key_name).as_oracle(compiler, connection)
        if not self.rhs:
            return (sql, params)
        lhs, lhs_params, _ = self.lhs.preprocess_lhs(compiler, connection)
        return ('(NOT %s OR %s IS NULL)' % (sql, lhs), tuple(params) + tuple(lhs_params))

    def as_sqlite(self, compiler, connection):
        template = 'JSON_TYPE(%s, %%s) IS NULL'
        if not self.rhs:
            template = 'JSON_TYPE(%s, %%s) IS NOT NULL'
        return HasKeyOrArrayIndex(self.lhs.lhs, self.lhs.key_name).as_sql(compiler, connection, template=template)