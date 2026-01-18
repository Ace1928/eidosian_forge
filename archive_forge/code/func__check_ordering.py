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
def _check_ordering(self, obj):
    """Check that ordering refers to existing fields or is random."""
    if obj.ordering is None:
        return []
    elif not isinstance(obj.ordering, (list, tuple)):
        return must_be('a list or tuple', option='ordering', obj=obj, id='admin.E031')
    else:
        return list(chain.from_iterable((self._check_ordering_item(obj, field_name, 'ordering[%d]' % index) for index, field_name in enumerate(obj.ordering))))