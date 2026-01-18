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
class Fieldline:

    def __init__(self, form, field, readonly_fields=None, model_admin=None):
        self.form = form
        if not hasattr(field, '__iter__') or isinstance(field, str):
            self.fields = [field]
        else:
            self.fields = field
        self.has_visible_field = not all((field in self.form.fields and self.form.fields[field].widget.is_hidden for field in self.fields))
        self.model_admin = model_admin
        if readonly_fields is None:
            readonly_fields = ()
        self.readonly_fields = readonly_fields

    def __iter__(self):
        for i, field in enumerate(self.fields):
            if field in self.readonly_fields:
                yield AdminReadonlyField(self.form, field, is_first=i == 0, model_admin=self.model_admin)
            else:
                yield AdminField(self.form, field, is_first=i == 0)

    def errors(self):
        return mark_safe('\n'.join((self.form[f].errors.as_ul() for f in self.fields if f not in self.readonly_fields)).strip('\n'))