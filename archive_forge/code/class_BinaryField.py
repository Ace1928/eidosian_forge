import copy
import datetime
import decimal
import operator
import uuid
import warnings
from base64 import b64decode, b64encode
from functools import partialmethod, total_ordering
from django import forms
from django.apps import apps
from django.conf import settings
from django.core import checks, exceptions, validators
from django.db import connection, connections, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import DeferredAttribute, RegisterLookupMixin
from django.utils import timezone
from django.utils.choices import (
from django.utils.datastructures import DictWrapper
from django.utils.dateparse import (
from django.utils.duration import duration_microseconds, duration_string
from django.utils.functional import Promise, cached_property
from django.utils.ipv6 import clean_ipv6_address
from django.utils.itercompat import is_iterable
from django.utils.text import capfirst
from django.utils.translation import gettext_lazy as _
class BinaryField(Field):
    description = _('Raw binary data')
    empty_values = [None, b'']

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('editable', False)
        super().__init__(*args, **kwargs)
        if self.max_length is not None:
            self.validators.append(validators.MaxLengthValidator(self.max_length))

    def check(self, **kwargs):
        return [*super().check(**kwargs), *self._check_str_default_value()]

    def _check_str_default_value(self):
        if self.has_default() and isinstance(self.default, str):
            return [checks.Error("BinaryField's default cannot be a string. Use bytes content instead.", obj=self, id='fields.E170')]
        return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.editable:
            kwargs['editable'] = True
        else:
            del kwargs['editable']
        return (name, path, args, kwargs)

    def get_internal_type(self):
        return 'BinaryField'

    def get_placeholder(self, value, compiler, connection):
        return connection.ops.binary_placeholder_sql(value)

    def get_default(self):
        if self.has_default() and (not callable(self.default)):
            return self.default
        default = super().get_default()
        if default == '':
            return b''
        return default

    def get_db_prep_value(self, value, connection, prepared=False):
        value = super().get_db_prep_value(value, connection, prepared)
        if value is not None:
            return connection.Database.Binary(value)
        return value

    def value_to_string(self, obj):
        """Binary data is serialized as base64"""
        return b64encode(self.value_from_object(obj)).decode('ascii')

    def to_python(self, value):
        if isinstance(value, str):
            return memoryview(b64decode(value.encode('ascii')))
        return value