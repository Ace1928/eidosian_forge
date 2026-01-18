import functools
import inspect
import warnings
from functools import partial
from django import forms
from django.apps import apps
from django.conf import SettingsReference, settings
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
from .related_lookups import (
from .reverse_related import ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel
def _check_ignored_options(self, **kwargs):
    warnings = []
    if self.has_null_arg:
        warnings.append(checks.Warning('null has no effect on ManyToManyField.', obj=self, id='fields.W340'))
    if self._validators:
        warnings.append(checks.Warning('ManyToManyField does not support validators.', obj=self, id='fields.W341'))
    if self.remote_field.symmetrical and self._related_name:
        warnings.append(checks.Warning('related_name has no effect on ManyToManyField with a symmetrical relationship, e.g. to "self".', obj=self, id='fields.W345'))
    if self.db_comment:
        warnings.append(checks.Warning('db_comment has no effect on ManyToManyField.', obj=self, id='fields.W346'))
    return warnings