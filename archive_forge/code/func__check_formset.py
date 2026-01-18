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
def _check_formset(self, obj):
    """Check formset is a subclass of BaseModelFormSet."""
    if not _issubclass(obj.formset, BaseModelFormSet):
        return must_inherit_from(parent='BaseModelFormSet', option='formset', obj=obj, id='admin.E206')
    else:
        return []