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
class InlineAdminFormSet:
    """
    A wrapper around an inline formset for use in the admin system.
    """

    def __init__(self, inline, formset, fieldsets, prepopulated_fields=None, readonly_fields=None, model_admin=None, has_add_permission=True, has_change_permission=True, has_delete_permission=True, has_view_permission=True):
        self.opts = inline
        self.formset = formset
        self.fieldsets = fieldsets
        self.model_admin = model_admin
        if readonly_fields is None:
            readonly_fields = ()
        self.readonly_fields = readonly_fields
        if prepopulated_fields is None:
            prepopulated_fields = {}
        self.prepopulated_fields = prepopulated_fields
        self.classes = ' '.join(inline.classes) if inline.classes else ''
        self.has_add_permission = has_add_permission
        self.has_change_permission = has_change_permission
        self.has_delete_permission = has_delete_permission
        self.has_view_permission = has_view_permission

    def __iter__(self):
        if self.has_change_permission:
            readonly_fields_for_editing = self.readonly_fields
        else:
            readonly_fields_for_editing = self.readonly_fields + flatten_fieldsets(self.fieldsets)
        for form, original in zip(self.formset.initial_forms, self.formset.get_queryset()):
            view_on_site_url = self.opts.get_view_on_site_url(original)
            yield InlineAdminForm(self.formset, form, self.fieldsets, self.prepopulated_fields, original, readonly_fields_for_editing, model_admin=self.opts, view_on_site_url=view_on_site_url)
        for form in self.formset.extra_forms:
            yield InlineAdminForm(self.formset, form, self.fieldsets, self.prepopulated_fields, None, self.readonly_fields, model_admin=self.opts)
        if self.has_add_permission:
            yield InlineAdminForm(self.formset, self.formset.empty_form, self.fieldsets, self.prepopulated_fields, None, self.readonly_fields, model_admin=self.opts)

    def fields(self):
        fk = getattr(self.formset, 'fk', None)
        empty_form = self.formset.empty_form
        meta_labels = empty_form._meta.labels or {}
        meta_help_texts = empty_form._meta.help_texts or {}
        for i, field_name in enumerate(flatten_fieldsets(self.fieldsets)):
            if fk and fk.name == field_name:
                continue
            if not self.has_change_permission or field_name in self.readonly_fields:
                form_field = empty_form.fields.get(field_name)
                widget_is_hidden = False
                if form_field is not None:
                    widget_is_hidden = form_field.widget.is_hidden
                yield {'name': field_name, 'label': meta_labels.get(field_name) or label_for_field(field_name, self.opts.model, self.opts, form=empty_form), 'widget': {'is_hidden': widget_is_hidden}, 'required': False, 'help_text': meta_help_texts.get(field_name) or help_text_for_field(field_name, self.opts.model)}
            else:
                form_field = empty_form.fields[field_name]
                label = form_field.label
                if label is None:
                    label = label_for_field(field_name, self.opts.model, self.opts, form=empty_form)
                yield {'name': field_name, 'label': label, 'widget': form_field.widget, 'required': form_field.required, 'help_text': form_field.help_text}

    def inline_formset_data(self):
        verbose_name = self.opts.verbose_name
        return json.dumps({'name': '#%s' % self.formset.prefix, 'options': {'prefix': self.formset.prefix, 'addText': gettext('Add another %(verbose_name)s') % {'verbose_name': capfirst(verbose_name)}, 'deleteText': gettext('Remove')}})

    @property
    def forms(self):
        return self.formset.forms

    def non_form_errors(self):
        return self.formset.non_form_errors()

    @property
    def is_bound(self):
        return self.formset.is_bound

    @property
    def total_form_count(self):
        return self.formset.total_form_count

    @property
    def media(self):
        media = self.opts.media + self.formset.media
        for fs in self:
            media += fs.media
        return media