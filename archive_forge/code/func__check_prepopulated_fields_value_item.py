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
def _check_prepopulated_fields_value_item(self, obj, field_name, label):
    """For `prepopulated_fields` equal to {"slug": ("title",)},
        `field_name` is "title"."""
    try:
        obj.model._meta.get_field(field_name)
    except FieldDoesNotExist:
        return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E030')
    else:
        return []