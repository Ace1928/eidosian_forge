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
def _check_action_permission_methods(self, obj):
    """
        Actions with an allowed_permission attribute require the ModelAdmin to
        implement a has_<perm>_permission() method for each permission.
        """
    actions = obj._get_base_actions()
    errors = []
    for func, name, _ in actions:
        if not hasattr(func, 'allowed_permissions'):
            continue
        for permission in func.allowed_permissions:
            method_name = 'has_%s_permission' % permission
            if not hasattr(obj, method_name):
                errors.append(checks.Error('%s must define a %s() method for the %s action.' % (obj.__class__.__name__, method_name, func.__name__), obj=obj.__class__, id='admin.E129'))
    return errors