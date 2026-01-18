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
class TimeField(DateTimeCheckMixin, Field):
    empty_strings_allowed = False
    default_error_messages = {'invalid': _('“%(value)s” value has an invalid format. It must be in HH:MM[:ss[.uuuuuu]] format.'), 'invalid_time': _('“%(value)s” value has the correct format (HH:MM[:ss[.uuuuuu]]) but it is an invalid time.')}
    description = _('Time')

    def __init__(self, verbose_name=None, name=None, auto_now=False, auto_now_add=False, **kwargs):
        self.auto_now, self.auto_now_add = (auto_now, auto_now_add)
        if auto_now or auto_now_add:
            kwargs['editable'] = False
            kwargs['blank'] = True
        super().__init__(verbose_name, name, **kwargs)

    def _check_fix_default_value(self):
        """
        Warn that using an actual date or datetime value is probably wrong;
        it's only evaluated on server startup.
        """
        if not self.has_default():
            return []
        value = self.default
        if isinstance(value, datetime.datetime):
            now = None
        elif isinstance(value, datetime.time):
            now = _get_naive_now()
            value = datetime.datetime.combine(now.date(), value)
        else:
            return []
        return self._check_if_value_fixed(value, now=now)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.auto_now is not False:
            kwargs['auto_now'] = self.auto_now
        if self.auto_now_add is not False:
            kwargs['auto_now_add'] = self.auto_now_add
        if self.auto_now or self.auto_now_add:
            del kwargs['blank']
            del kwargs['editable']
        return (name, path, args, kwargs)

    def get_internal_type(self):
        return 'TimeField'

    def to_python(self, value):
        if value is None:
            return None
        if isinstance(value, datetime.time):
            return value
        if isinstance(value, datetime.datetime):
            return value.time()
        try:
            parsed = parse_time(value)
            if parsed is not None:
                return parsed
        except ValueError:
            raise exceptions.ValidationError(self.error_messages['invalid_time'], code='invalid_time', params={'value': value})
        raise exceptions.ValidationError(self.error_messages['invalid'], code='invalid', params={'value': value})

    def pre_save(self, model_instance, add):
        if self.auto_now or (self.auto_now_add and add):
            value = datetime.datetime.now().time()
            setattr(model_instance, self.attname, value)
            return value
        else:
            return super().pre_save(model_instance, add)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def get_db_prep_value(self, value, connection, prepared=False):
        if not prepared:
            value = self.get_prep_value(value)
        return connection.ops.adapt_timefield_value(value)

    def value_to_string(self, obj):
        val = self.value_from_object(obj)
        return '' if val is None else val.isoformat()

    def formfield(self, **kwargs):
        return super().formfield(**{'form_class': forms.TimeField, **kwargs})