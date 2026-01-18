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
class CharField(Field):

    def __init__(self, *args, db_collation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_collation = db_collation
        if self.max_length is not None:
            self.validators.append(validators.MaxLengthValidator(self.max_length))

    @property
    def description(self):
        if self.max_length is not None:
            return _('String (up to %(max_length)s)')
        else:
            return _('String (unlimited)')

    def check(self, **kwargs):
        databases = kwargs.get('databases') or []
        return [*super().check(**kwargs), *self._check_db_collation(databases), *self._check_max_length_attribute(**kwargs)]

    def _check_max_length_attribute(self, **kwargs):
        if self.max_length is None:
            if connection.features.supports_unlimited_charfield or 'supports_unlimited_charfield' in self.model._meta.required_db_features:
                return []
            return [checks.Error("CharFields must define a 'max_length' attribute.", obj=self, id='fields.E120')]
        elif not isinstance(self.max_length, int) or isinstance(self.max_length, bool) or self.max_length <= 0:
            return [checks.Error("'max_length' must be a positive integer.", obj=self, id='fields.E121')]
        else:
            return []

    def _check_db_collation(self, databases):
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, self.model):
                continue
            connection = connections[db]
            if not (self.db_collation is None or 'supports_collation_on_charfield' in self.model._meta.required_db_features or connection.features.supports_collation_on_charfield):
                errors.append(checks.Error('%s does not support a database collation on CharFields.' % connection.display_name, obj=self, id='fields.E190'))
        return errors

    def cast_db_type(self, connection):
        if self.max_length is None:
            return connection.ops.cast_char_field_without_max_length
        return super().cast_db_type(connection)

    def db_parameters(self, connection):
        db_params = super().db_parameters(connection)
        db_params['collation'] = self.db_collation
        return db_params

    def get_internal_type(self):
        return 'CharField'

    def to_python(self, value):
        if isinstance(value, str) or value is None:
            return value
        return str(value)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def formfield(self, **kwargs):
        defaults = {'max_length': self.max_length}
        if self.null and (not connection.features.interprets_empty_strings_as_nulls):
            defaults['empty_value'] = None
        defaults.update(kwargs)
        return super().formfield(**defaults)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.db_collation:
            kwargs['db_collation'] = self.db_collation
        return (name, path, args, kwargs)