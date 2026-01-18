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
def _check_table_uniqueness(self, **kwargs):
    if isinstance(self.remote_field.through, str) or not self.remote_field.through._meta.managed:
        return []
    registered_tables = {model._meta.db_table: model for model in self.opts.apps.get_models(include_auto_created=True) if model != self.remote_field.through and model._meta.managed}
    m2m_db_table = self.m2m_db_table()
    model = registered_tables.get(m2m_db_table)
    if model and model._meta.concrete_model != self.remote_field.through._meta.concrete_model:
        if model._meta.auto_created:

            def _get_field_name(model):
                for field in model._meta.auto_created._meta.many_to_many:
                    if field.remote_field.through is model:
                        return field.name
            opts = model._meta.auto_created._meta
            clashing_obj = '%s.%s' % (opts.label, _get_field_name(model))
        else:
            clashing_obj = model._meta.label
        if settings.DATABASE_ROUTERS:
            error_class, error_id = (checks.Warning, 'fields.W344')
            error_hint = 'You have configured settings.DATABASE_ROUTERS. Verify that the table of %r is correctly routed to a separate database.' % clashing_obj
        else:
            error_class, error_id = (checks.Error, 'fields.E340')
            error_hint = None
        return [error_class("The field's intermediary table '%s' clashes with the table name of '%s'." % (m2m_db_table, clashing_obj), obj=self, hint=error_hint, id=error_id)]
    return []