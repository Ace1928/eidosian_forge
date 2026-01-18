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
@classmethod
def _check_field_name_clashes(cls):
    """Forbid field shadowing in multi-table inheritance."""
    errors = []
    used_fields = {}
    for parent in cls._meta.get_parent_list():
        for f in parent._meta.local_fields:
            clash = used_fields.get(f.name) or used_fields.get(f.attname) or None
            if clash:
                errors.append(checks.Error("The field '%s' from parent model '%s' clashes with the field '%s' from parent model '%s'." % (clash.name, clash.model._meta, f.name, f.model._meta), obj=cls, id='models.E005'))
            used_fields[f.name] = f
            used_fields[f.attname] = f
    for parent in cls._meta.get_parent_list():
        for f in parent._meta.get_fields():
            if f not in used_fields:
                used_fields[f.name] = f
    for parent_link in cls._meta.parents.values():
        if not parent_link:
            continue
        clash = used_fields.get(parent_link.name) or None
        if clash:
            errors.append(checks.Error(f"The field '{parent_link.name}' clashes with the field '{clash.name}' from model '{clash.model._meta}'.", obj=cls, id='models.E006'))
    for f in cls._meta.local_fields:
        clash = used_fields.get(f.name) or used_fields.get(f.attname) or None
        id_conflict = f.name == 'id' and clash and (clash.name == 'id') and (clash.model == cls)
        if clash and (not id_conflict):
            errors.append(checks.Error("The field '%s' clashes with the field '%s' from model '%s'." % (f.name, clash.name, clash.model._meta), obj=f, id='models.E006'))
        used_fields[f.name] = f
        used_fields[f.attname] = f
    return errors