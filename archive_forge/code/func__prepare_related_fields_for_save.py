import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain
from asgiref.sync import sync_to_async
import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
from django.db import (
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.expressions import DatabaseDefault, RawSQL
from django.db.models.fields.related import (
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
from django.db.models.utils import AltersData, make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
def _prepare_related_fields_for_save(self, operation_name, fields=None):
    for field in self._meta.concrete_fields:
        if fields and field not in fields:
            continue
        if field.is_relation and field.is_cached(self):
            obj = getattr(self, field.name, None)
            if not obj:
                continue
            if obj.pk is None:
                if not field.remote_field.multiple:
                    field.remote_field.delete_cached_value(obj)
                raise ValueError("%s() prohibited to prevent data loss due to unsaved related object '%s'." % (operation_name, field.name))
            elif getattr(self, field.attname) in field.empty_values:
                setattr(self, field.name, obj)
            if getattr(obj, field.target_field.attname) != getattr(self, field.attname):
                field.delete_cached_value(self)
    for field in self._meta.private_fields:
        if fields and field not in fields:
            continue
        if field.is_relation and field.is_cached(self) and hasattr(field, 'fk_field'):
            obj = field.get_cached_value(self, default=None)
            if obj and obj.pk is None:
                raise ValueError(f"{operation_name}() prohibited to prevent data loss due to unsaved related object '{field.name}'.")