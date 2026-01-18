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
def _check_filter_item(self, obj, field_name, label):
    """Check one item of `filter_vertical` or `filter_horizontal`, i.e.
        check that given field exists and is a ManyToManyField."""
    try:
        field = obj.model._meta.get_field(field_name)
    except FieldDoesNotExist:
        return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E019')
    else:
        if not field.many_to_many or isinstance(field, models.ManyToManyRel):
            return must_be('a many-to-many field', option=label, obj=obj, id='admin.E020')
        elif not field.remote_field.through._meta.auto_created:
            return [checks.Error(f"The value of '{label}' cannot include the ManyToManyField '{field_name}', because that field manually specifies a relationship model.", obj=obj.__class__, id='admin.E013')]
        else:
            return []