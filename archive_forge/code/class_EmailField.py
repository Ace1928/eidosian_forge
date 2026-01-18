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
class EmailField(CharField):
    default_validators = [validators.validate_email]
    description = _('Email address')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('max_length', 254)
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        return (name, path, args, kwargs)

    def formfield(self, **kwargs):
        return super().formfield(**{'form_class': forms.EmailField, **kwargs})