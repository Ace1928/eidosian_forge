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
class InlineModelAdminChecks(BaseModelAdminChecks):

    def check(self, inline_obj, **kwargs):
        parent_model = inline_obj.parent_model
        return [*super().check(inline_obj), *self._check_relation(inline_obj, parent_model), *self._check_exclude_of_parent_model(inline_obj, parent_model), *self._check_extra(inline_obj), *self._check_max_num(inline_obj), *self._check_min_num(inline_obj), *self._check_formset(inline_obj)]

    def _check_exclude_of_parent_model(self, obj, parent_model):
        errors = super()._check_exclude(obj)
        if errors:
            return []
        if self._check_relation(obj, parent_model):
            return []
        if obj.exclude is None:
            return []
        fk = _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        if fk.name in obj.exclude:
            return [checks.Error("Cannot exclude the field '%s', because it is the foreign key to the parent model '%s'." % (fk.name, parent_model._meta.label), obj=obj.__class__, id='admin.E201')]
        else:
            return []

    def _check_relation(self, obj, parent_model):
        try:
            _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        except ValueError as e:
            return [checks.Error(e.args[0], obj=obj.__class__, id='admin.E202')]
        else:
            return []

    def _check_extra(self, obj):
        """Check that extra is an integer."""
        if not isinstance(obj.extra, int):
            return must_be('an integer', option='extra', obj=obj, id='admin.E203')
        else:
            return []

    def _check_max_num(self, obj):
        """Check that max_num is an integer."""
        if obj.max_num is None:
            return []
        elif not isinstance(obj.max_num, int):
            return must_be('an integer', option='max_num', obj=obj, id='admin.E204')
        else:
            return []

    def _check_min_num(self, obj):
        """Check that min_num is an integer."""
        if obj.min_num is None:
            return []
        elif not isinstance(obj.min_num, int):
            return must_be('an integer', option='min_num', obj=obj, id='admin.E205')
        else:
            return []

    def _check_formset(self, obj):
        """Check formset is a subclass of BaseModelFormSet."""
        if not _issubclass(obj.formset, BaseModelFormSet):
            return must_inherit_from(parent='BaseModelFormSet', option='formset', obj=obj, id='admin.E206')
        else:
            return []