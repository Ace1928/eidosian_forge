from functools import partial
from django.contrib.admin.checks import InlineModelAdminChecks
from django.contrib.admin.options import InlineModelAdmin, flatten_fieldsets
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.forms import (
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.forms import ALL_FIELDS
from django.forms.models import modelform_defines_fields
class GenericInlineModelAdminChecks(InlineModelAdminChecks):

    def _check_exclude_of_parent_model(self, obj, parent_model):
        return []

    def _check_relation(self, obj, parent_model):
        gfks = [f for f in obj.model._meta.private_fields if isinstance(f, GenericForeignKey)]
        if not gfks:
            return [checks.Error("'%s' has no GenericForeignKey." % obj.model._meta.label, obj=obj.__class__, id='admin.E301')]
        else:
            try:
                obj.model._meta.get_field(obj.ct_field)
            except FieldDoesNotExist:
                return [checks.Error("'ct_field' references '%s', which is not a field on '%s'." % (obj.ct_field, obj.model._meta.label), obj=obj.__class__, id='admin.E302')]
            try:
                obj.model._meta.get_field(obj.ct_fk_field)
            except FieldDoesNotExist:
                return [checks.Error("'ct_fk_field' references '%s', which is not a field on '%s'." % (obj.ct_fk_field, obj.model._meta.label), obj=obj.__class__, id='admin.E303')]
            for gfk in gfks:
                if gfk.ct_field == obj.ct_field and gfk.fk_field == obj.ct_fk_field:
                    return []
            return [checks.Error("'%s' has no GenericForeignKey using content type field '%s' and object ID field '%s'." % (obj.model._meta.label, obj.ct_field, obj.ct_fk_field), obj=obj.__class__, id='admin.E304')]