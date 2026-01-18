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
def _check_clashes(self):
    """Check accessor and reverse query name clashes."""
    from django.db.models.base import ModelBase
    errors = []
    opts = self.model._meta
    if not isinstance(self.remote_field.model, ModelBase):
        return []
    rel_opts = self.remote_field.model._meta
    rel_is_hidden = self.remote_field.is_hidden()
    rel_name = self.remote_field.get_accessor_name()
    rel_query_name = self.related_query_name()
    field_name = '%s.%s' % (opts.label, self.name)
    potential_clashes = rel_opts.fields + rel_opts.many_to_many
    for clash_field in potential_clashes:
        clash_name = '%s.%s' % (rel_opts.label, clash_field.name)
        if not rel_is_hidden and clash_field.name == rel_name:
            errors.append(checks.Error(f"Reverse accessor '{rel_opts.object_name}.{rel_name}' for '{field_name}' clashes with field name '{clash_name}'.", hint="Rename field '%s', or add/change a related_name argument to the definition for field '%s'." % (clash_name, field_name), obj=self, id='fields.E302'))
        if clash_field.name == rel_query_name:
            errors.append(checks.Error("Reverse query name for '%s' clashes with field name '%s'." % (field_name, clash_name), hint="Rename field '%s', or add/change a related_name argument to the definition for field '%s'." % (clash_name, field_name), obj=self, id='fields.E303'))
    potential_clashes = (r for r in rel_opts.related_objects if r.field is not self)
    for clash_field in potential_clashes:
        clash_name = '%s.%s' % (clash_field.related_model._meta.label, clash_field.field.name)
        if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
            errors.append(checks.Error(f"Reverse accessor '{rel_opts.object_name}.{rel_name}' for '{field_name}' clashes with reverse accessor for '{clash_name}'.", hint="Add or change a related_name argument to the definition for '%s' or '%s'." % (field_name, clash_name), obj=self, id='fields.E304'))
        if clash_field.get_accessor_name() == rel_query_name:
            errors.append(checks.Error("Reverse query name for '%s' clashes with reverse query name for '%s'." % (field_name, clash_name), hint="Add or change a related_name argument to the definition for '%s' or '%s'." % (field_name, clash_name), obj=self, id='fields.E305'))
    return errors