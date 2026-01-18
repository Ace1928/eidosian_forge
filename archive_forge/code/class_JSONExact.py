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
class JSONExact(lookups.Exact):
    can_use_none_as_rhs = True

    def process_rhs(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if rhs == '%s' and rhs_params == [None]:
            rhs_params = ['null']
        if connection.vendor == 'mysql':
            func = ["JSON_EXTRACT(%s, '$')"] * len(rhs_params)
            rhs %= tuple(func)
        return (rhs, rhs_params)