import collections
from itertools import chain
from django.apps import apps
from django.conf import settings
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.utils import NotRelationField, flatten, get_fields_from_path
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import Combinable
from django.forms.models import BaseModelForm, BaseModelFormSet, _get_foreign_key
from django.template import engines
from django.template.backends.django import DjangoTemplates
from django.utils.module_loading import import_string
def _check_fields(self, obj):
    """Check that `fields` only refer to existing fields, doesn't contain
        duplicates. Check if at most one of `fields` and `fieldsets` is defined.
        """
    if obj.fields is None:
        return []
    elif not isinstance(obj.fields, (list, tuple)):
        return must_be('a list or tuple', option='fields', obj=obj, id='admin.E004')
    elif obj.fieldsets:
        return [checks.Error("Both 'fieldsets' and 'fields' are specified.", obj=obj.__class__, id='admin.E005')]
    fields = flatten(obj.fields)
    if len(fields) != len(set(fields)):
        return [checks.Error("The value of 'fields' contains duplicate field(s).", obj=obj.__class__, id='admin.E006')]
    return list(chain.from_iterable((self._check_field_spec(obj, field_name, 'fields') for field_name in obj.fields)))