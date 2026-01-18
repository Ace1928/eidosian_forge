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
def _check_date_hierarchy(self, obj):
    """Check that date_hierarchy refers to DateField or DateTimeField."""
    if obj.date_hierarchy is None:
        return []
    else:
        try:
            field = get_fields_from_path(obj.model, obj.date_hierarchy)[-1]
        except (NotRelationField, FieldDoesNotExist):
            return [checks.Error("The value of 'date_hierarchy' refers to '%s', which does not refer to a Field." % obj.date_hierarchy, obj=obj.__class__, id='admin.E127')]
        else:
            if not isinstance(field, (models.DateField, models.DateTimeField)):
                return must_be('a DateField or DateTimeField', option='date_hierarchy', obj=obj, id='admin.E128')
            else:
                return []