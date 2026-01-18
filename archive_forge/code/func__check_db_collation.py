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
def _check_db_collation(self, databases):
    errors = []
    for db in databases:
        if not router.allow_migrate_model(db, self.model):
            continue
        connection = connections[db]
        if not (self.db_collation is None or 'supports_collation_on_textfield' in self.model._meta.required_db_features or connection.features.supports_collation_on_textfield):
            errors.append(checks.Error('%s does not support a database collation on TextFields.' % connection.display_name, obj=self, id='fields.E190'))
    return errors