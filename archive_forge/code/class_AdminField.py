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
class AdminField:

    def __init__(self, form, field, is_first):
        self.field = form[field]
        self.is_first = is_first
        self.is_checkbox = isinstance(self.field.field.widget, forms.CheckboxInput)
        self.is_readonly = False

    def label_tag(self):
        classes = []
        contents = conditional_escape(self.field.label)
        if self.is_checkbox:
            classes.append('vCheckboxLabel')
        if self.field.field.required:
            classes.append('required')
        if not self.is_first:
            classes.append('inline')
        attrs = {'class': ' '.join(classes)} if classes else {}
        return self.field.label_tag(contents=mark_safe(contents), attrs=attrs, label_suffix='' if self.is_checkbox else None)

    def errors(self):
        return mark_safe(self.field.errors.as_ul())