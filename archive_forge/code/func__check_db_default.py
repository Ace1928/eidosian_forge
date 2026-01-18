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
def _check_db_default(self, databases=None, **kwargs):
    from django.db.models.expressions import Value
    if self.db_default is NOT_PROVIDED or isinstance(self.db_default, Value) or databases is None:
        return []
    errors = []
    for db in databases:
        if not router.allow_migrate_model(db, self.model):
            continue
        connection = connections[db]
        if not getattr(self._db_default_expression, 'allowed_default', False) and connection.features.supports_expression_defaults:
            msg = f'{self.db_default} cannot be used in db_default.'
            errors.append(checks.Error(msg, obj=self, id='fields.E012'))
        if not (connection.features.supports_expression_defaults or 'supports_expression_defaults' in self.model._meta.required_db_features):
            msg = f'{connection.display_name} does not support default database values with expressions (db_default).'
            errors.append(checks.Error(msg, obj=self, id='fields.E011'))
    return errors