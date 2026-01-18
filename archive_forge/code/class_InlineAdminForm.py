import json
from django import forms
from django.contrib.admin.utils import (
from django.core.exceptions import ObjectDoesNotExist
from django.db.models.fields.related import (
from django.forms.utils import flatatt
from django.template.defaultfilters import capfirst, linebreaksbr
from django.urls import NoReverseMatch, reverse
from django.utils.html import conditional_escape, format_html
from django.utils.safestring import mark_safe
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
class InlineAdminForm(AdminForm):
    """
    A wrapper around an inline form for use in the admin system.
    """

    def __init__(self, formset, form, fieldsets, prepopulated_fields, original, readonly_fields=None, model_admin=None, view_on_site_url=None):
        self.formset = formset
        self.model_admin = model_admin
        self.original = original
        self.show_url = original and view_on_site_url is not None
        self.absolute_url = view_on_site_url
        super().__init__(form, fieldsets, prepopulated_fields, readonly_fields, model_admin)

    def __iter__(self):
        for name, options in self.fieldsets:
            yield InlineFieldset(self.formset, self.form, name, self.readonly_fields, model_admin=self.model_admin, **options)

    def needs_explicit_pk_field(self):
        return self.form._meta.model._meta.auto_field or not self.form._meta.model._meta.pk.editable or any((parent._meta.auto_field or not parent._meta.model._meta.pk.editable for parent in self.form._meta.model._meta.get_parent_list()))

    def pk_field(self):
        return AdminField(self.form, self.formset._pk_field.name, False)

    def fk_field(self):
        fk = getattr(self.formset, 'fk', None)
        if fk:
            return AdminField(self.form, fk.name, False)
        else:
            return ''

    def deletion_field(self):
        from django.forms.formsets import DELETION_FIELD_NAME
        return AdminField(self.form, DELETION_FIELD_NAME, False)