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
def _check_prepopulated_fields_key(self, obj, field_name, label):
    """Check a key of `prepopulated_fields` dictionary, i.e. check that it
        is a name of existing field and the field is one of the allowed types.
        """
    try:
        field = obj.model._meta.get_field(field_name)
    except FieldDoesNotExist:
        return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E027')
    else:
        if isinstance(field, (models.DateTimeField, models.ForeignKey, models.ManyToManyField)):
            return [checks.Error("The value of '%s' refers to '%s', which must not be a DateTimeField, a ForeignKey, a OneToOneField, or a ManyToManyField." % (label, field_name), obj=obj.__class__, id='admin.E028')]
        else:
            return []