import copy
import enum
import json
import re
from functools import partial, update_wrapper
from urllib.parse import parse_qsl
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import helpers, widgets
from django.contrib.admin.checks import (
from django.contrib.admin.exceptions import DisallowedModelAdminToField, NotRegistered
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
from django.contrib.admin.widgets import AutocompleteSelect, AutocompleteSelectMultiple
from django.contrib.auth import get_permission_codename
from django.core.exceptions import (
from django.core.paginator import Paginator
from django.db import models, router, transaction
from django.db.models.constants import LOOKUP_SEP
from django.forms.formsets import DELETION_FIELD_NAME, all_valid
from django.forms.models import (
from django.forms.widgets import CheckboxSelectMultiple, SelectMultiple
from django.http import HttpResponseRedirect
from django.http.response import HttpResponseBase
from django.template.response import SimpleTemplateResponse, TemplateResponse
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.html import format_html
from django.utils.http import urlencode
from django.utils.safestring import mark_safe
from django.utils.text import (
from django.utils.translation import gettext as _
from django.utils.translation import ngettext
from django.views.decorators.csrf import csrf_protect
from django.views.generic import RedirectView
def _changeform_view(self, request, object_id, form_url, extra_context):
    to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
    if to_field and (not self.to_field_allowed(request, to_field)):
        raise DisallowedModelAdminToField('The field %s cannot be referenced.' % to_field)
    if request.method == 'POST' and '_saveasnew' in request.POST:
        object_id = None
    add = object_id is None
    if add:
        if not self.has_add_permission(request):
            raise PermissionDenied
        obj = None
    else:
        obj = self.get_object(request, unquote(object_id), to_field)
        if request.method == 'POST':
            if not self.has_change_permission(request, obj):
                raise PermissionDenied
        elif not self.has_view_or_change_permission(request, obj):
            raise PermissionDenied
        if obj is None:
            return self._get_obj_does_not_exist_redirect(request, self.opts, object_id)
    fieldsets = self.get_fieldsets(request, obj)
    ModelForm = self.get_form(request, obj, change=not add, fields=flatten_fieldsets(fieldsets))
    if request.method == 'POST':
        form = ModelForm(request.POST, request.FILES, instance=obj)
        formsets, inline_instances = self._create_formsets(request, form.instance, change=not add)
        form_validated = form.is_valid()
        if form_validated:
            new_object = self.save_form(request, form, change=not add)
        else:
            new_object = form.instance
        if all_valid(formsets) and form_validated:
            self.save_model(request, new_object, form, not add)
            self.save_related(request, form, formsets, not add)
            change_message = self.construct_change_message(request, form, formsets, add)
            if add:
                self.log_addition(request, new_object, change_message)
                return self.response_add(request, new_object)
            else:
                self.log_change(request, new_object, change_message)
                return self.response_change(request, new_object)
        else:
            form_validated = False
    elif add:
        initial = self.get_changeform_initial_data(request)
        form = ModelForm(initial=initial)
        formsets, inline_instances = self._create_formsets(request, form.instance, change=False)
    else:
        form = ModelForm(instance=obj)
        formsets, inline_instances = self._create_formsets(request, obj, change=True)
    if not add and (not self.has_change_permission(request, obj)):
        readonly_fields = flatten_fieldsets(fieldsets)
    else:
        readonly_fields = self.get_readonly_fields(request, obj)
    admin_form = helpers.AdminForm(form, list(fieldsets), self.get_prepopulated_fields(request, obj) if add or self.has_change_permission(request, obj) else {}, readonly_fields, model_admin=self)
    media = self.media + admin_form.media
    inline_formsets = self.get_inline_formsets(request, formsets, inline_instances, obj)
    for inline_formset in inline_formsets:
        media += inline_formset.media
    if add:
        title = _('Add %s')
    elif self.has_change_permission(request, obj):
        title = _('Change %s')
    else:
        title = _('View %s')
    context = {**self.admin_site.each_context(request), 'title': title % self.opts.verbose_name, 'subtitle': str(obj) if obj else None, 'adminform': admin_form, 'object_id': object_id, 'original': obj, 'is_popup': IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET, 'to_field': to_field, 'media': media, 'inline_admin_formsets': inline_formsets, 'errors': helpers.AdminErrorList(form, formsets), 'preserved_filters': self.get_preserved_filters(request)}
    if request.method == 'POST' and (not form_validated) and ('_saveasnew' in request.POST):
        context['show_save'] = False
        context['show_save_and_continue'] = False
        add = False
    context.update(extra_context or {})
    return self.render_change_form(request, context, add=add, change=not add, obj=obj, form_url=form_url)