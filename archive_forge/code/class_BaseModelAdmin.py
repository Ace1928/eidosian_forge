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
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):
    """Functionality common to both ModelAdmin and InlineAdmin."""
    autocomplete_fields = ()
    raw_id_fields = ()
    fields = None
    exclude = None
    fieldsets = None
    form = forms.ModelForm
    filter_vertical = ()
    filter_horizontal = ()
    radio_fields = {}
    prepopulated_fields = {}
    formfield_overrides = {}
    readonly_fields = ()
    ordering = None
    sortable_by = None
    view_on_site = True
    show_full_result_count = True
    checks_class = BaseModelAdminChecks

    def check(self, **kwargs):
        return self.checks_class().check(self, **kwargs)

    def __init__(self):
        overrides = copy.deepcopy(FORMFIELD_FOR_DBFIELD_DEFAULTS)
        for k, v in self.formfield_overrides.items():
            overrides.setdefault(k, {}).update(v)
        self.formfield_overrides = overrides

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        """
        Hook for specifying the form Field instance for a given database Field
        instance.

        If kwargs are given, they're passed to the form Field's constructor.
        """
        if db_field.choices:
            return self.formfield_for_choice_field(db_field, request, **kwargs)
        if isinstance(db_field, (models.ForeignKey, models.ManyToManyField)):
            if db_field.__class__ in self.formfield_overrides:
                kwargs = {**self.formfield_overrides[db_field.__class__], **kwargs}
            if isinstance(db_field, models.ForeignKey):
                formfield = self.formfield_for_foreignkey(db_field, request, **kwargs)
            elif isinstance(db_field, models.ManyToManyField):
                formfield = self.formfield_for_manytomany(db_field, request, **kwargs)
            if formfield and db_field.name not in self.raw_id_fields:
                try:
                    related_modeladmin = self.admin_site.get_model_admin(db_field.remote_field.model)
                except NotRegistered:
                    wrapper_kwargs = {}
                else:
                    wrapper_kwargs = {'can_add_related': related_modeladmin.has_add_permission(request), 'can_change_related': related_modeladmin.has_change_permission(request), 'can_delete_related': related_modeladmin.has_delete_permission(request), 'can_view_related': related_modeladmin.has_view_permission(request)}
                formfield.widget = widgets.RelatedFieldWidgetWrapper(formfield.widget, db_field.remote_field, self.admin_site, **wrapper_kwargs)
            return formfield
        for klass in db_field.__class__.mro():
            if klass in self.formfield_overrides:
                kwargs = {**copy.deepcopy(self.formfield_overrides[klass]), **kwargs}
                return db_field.formfield(**kwargs)
        return db_field.formfield(**kwargs)

    def formfield_for_choice_field(self, db_field, request, **kwargs):
        """
        Get a form Field for a database Field that has declared choices.
        """
        if db_field.name in self.radio_fields:
            if 'widget' not in kwargs:
                kwargs['widget'] = widgets.AdminRadioSelect(attrs={'class': get_ul_class(self.radio_fields[db_field.name])})
            if 'choices' not in kwargs:
                kwargs['choices'] = db_field.get_choices(include_blank=db_field.blank, blank_choice=[('', _('None'))])
        return db_field.formfield(**kwargs)

    def get_field_queryset(self, db, db_field, request):
        """
        If the ModelAdmin specifies ordering, the queryset should respect that
        ordering.  Otherwise don't specify the queryset, let the field decide
        (return None in that case).
        """
        try:
            related_admin = self.admin_site.get_model_admin(db_field.remote_field.model)
        except NotRegistered:
            return None
        else:
            ordering = related_admin.get_ordering(request)
            if ordering is not None and ordering != ():
                return db_field.remote_field.model._default_manager.using(db).order_by(*ordering)
        return None

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        """
        Get a form Field for a ForeignKey.
        """
        db = kwargs.get('using')
        if 'widget' not in kwargs:
            if db_field.name in self.get_autocomplete_fields(request):
                kwargs['widget'] = AutocompleteSelect(db_field, self.admin_site, using=db)
            elif db_field.name in self.raw_id_fields:
                kwargs['widget'] = widgets.ForeignKeyRawIdWidget(db_field.remote_field, self.admin_site, using=db)
            elif db_field.name in self.radio_fields:
                kwargs['widget'] = widgets.AdminRadioSelect(attrs={'class': get_ul_class(self.radio_fields[db_field.name])})
                kwargs['empty_label'] = kwargs.get('empty_label', _('None')) if db_field.blank else None
        if 'queryset' not in kwargs:
            queryset = self.get_field_queryset(db, db_field, request)
            if queryset is not None:
                kwargs['queryset'] = queryset
        return db_field.formfield(**kwargs)

    def formfield_for_manytomany(self, db_field, request, **kwargs):
        """
        Get a form Field for a ManyToManyField.
        """
        if not db_field.remote_field.through._meta.auto_created:
            return None
        db = kwargs.get('using')
        if 'widget' not in kwargs:
            autocomplete_fields = self.get_autocomplete_fields(request)
            if db_field.name in autocomplete_fields:
                kwargs['widget'] = AutocompleteSelectMultiple(db_field, self.admin_site, using=db)
            elif db_field.name in self.raw_id_fields:
                kwargs['widget'] = widgets.ManyToManyRawIdWidget(db_field.remote_field, self.admin_site, using=db)
            elif db_field.name in [*self.filter_vertical, *self.filter_horizontal]:
                kwargs['widget'] = widgets.FilteredSelectMultiple(db_field.verbose_name, db_field.name in self.filter_vertical)
        if 'queryset' not in kwargs:
            queryset = self.get_field_queryset(db, db_field, request)
            if queryset is not None:
                kwargs['queryset'] = queryset
        form_field = db_field.formfield(**kwargs)
        if isinstance(form_field.widget, SelectMultiple) and form_field.widget.allow_multiple_selected and (not isinstance(form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple))):
            msg = _('Hold down “Control”, or “Command” on a Mac, to select more than one.')
            help_text = form_field.help_text
            form_field.help_text = format_lazy('{} {}', help_text, msg) if help_text else msg
        return form_field

    def get_autocomplete_fields(self, request):
        """
        Return a list of ForeignKey and/or ManyToMany fields which should use
        an autocomplete widget.
        """
        return self.autocomplete_fields

    def get_view_on_site_url(self, obj=None):
        if obj is None or not self.view_on_site:
            return None
        if callable(self.view_on_site):
            return self.view_on_site(obj)
        elif hasattr(obj, 'get_absolute_url'):
            return reverse('admin:view_on_site', kwargs={'content_type_id': get_content_type_for_model(obj).pk, 'object_id': obj.pk}, current_app=self.admin_site.name)

    def get_empty_value_display(self):
        """
        Return the empty_value_display set on ModelAdmin or AdminSite.
        """
        try:
            return mark_safe(self.empty_value_display)
        except AttributeError:
            return mark_safe(self.admin_site.empty_value_display)

    def get_exclude(self, request, obj=None):
        """
        Hook for specifying exclude.
        """
        return self.exclude

    def get_fields(self, request, obj=None):
        """
        Hook for specifying fields.
        """
        if self.fields:
            return self.fields
        form = self._get_form_for_get_fields(request, obj)
        return [*form.base_fields, *self.get_readonly_fields(request, obj)]

    def get_fieldsets(self, request, obj=None):
        """
        Hook for specifying fieldsets.
        """
        if self.fieldsets:
            return self.fieldsets
        return [(None, {'fields': self.get_fields(request, obj)})]

    def get_inlines(self, request, obj):
        """Hook for specifying custom inlines."""
        return self.inlines

    def get_ordering(self, request):
        """
        Hook for specifying field ordering.
        """
        return self.ordering or ()

    def get_readonly_fields(self, request, obj=None):
        """
        Hook for specifying custom readonly fields.
        """
        return self.readonly_fields

    def get_prepopulated_fields(self, request, obj=None):
        """
        Hook for specifying custom prepopulated fields.
        """
        return self.prepopulated_fields

    def get_queryset(self, request):
        """
        Return a QuerySet of all model instances that can be edited by the
        admin site. This is used by changelist_view.
        """
        qs = self.model._default_manager.get_queryset()
        ordering = self.get_ordering(request)
        if ordering:
            qs = qs.order_by(*ordering)
        return qs

    def get_sortable_by(self, request):
        """Hook for specifying which fields can be sorted in the changelist."""
        return self.sortable_by if self.sortable_by is not None else self.get_list_display(request)

    def lookup_allowed(self, lookup, value, request=None):
        from django.contrib.admin.filters import SimpleListFilter
        model = self.model
        for fk_lookup in model._meta.related_fkey_lookups:
            if callable(fk_lookup):
                fk_lookup = fk_lookup()
            if (lookup, value) in widgets.url_params_from_lookup_dict(fk_lookup).items():
                return True
        relation_parts = []
        prev_field = None
        parts = lookup.split(LOOKUP_SEP)
        for part in parts:
            try:
                field = model._meta.get_field(part)
            except FieldDoesNotExist:
                break
            if not prev_field or (prev_field.is_relation and field not in model._meta.parents.values() and (field is not model._meta.auto_field) and (model._meta.auto_field is None or part not in getattr(prev_field, 'to_fields', [])) and (field.is_relation or not field.primary_key)):
                relation_parts.append(part)
            if not getattr(field, 'path_infos', None):
                break
            prev_field = field
            model = field.path_infos[-1].to_opts.model
        if len(relation_parts) <= 1:
            return True
        valid_lookups = {self.date_hierarchy}
        list_filter = self.get_list_filter(request) if request is not None else self.list_filter
        for filter_item in list_filter:
            if isinstance(filter_item, type) and issubclass(filter_item, SimpleListFilter):
                valid_lookups.add(filter_item.parameter_name)
            elif isinstance(filter_item, (list, tuple)):
                valid_lookups.add(filter_item[0])
            else:
                valid_lookups.add(filter_item)
        return not {LOOKUP_SEP.join(relation_parts), LOOKUP_SEP.join(relation_parts + [part])}.isdisjoint(valid_lookups)

    def to_field_allowed(self, request, to_field):
        """
        Return True if the model associated with this admin should be
        allowed to be referenced by the specified field.
        """
        try:
            field = self.opts.get_field(to_field)
        except FieldDoesNotExist:
            return False
        if field.primary_key:
            return True
        for many_to_many in self.opts.many_to_many:
            if many_to_many.m2m_target_field_name() == to_field:
                return True
        registered_models = set()
        for model, admin in self.admin_site._registry.items():
            registered_models.add(model)
            for inline in admin.inlines:
                registered_models.add(inline.model)
        related_objects = (f for f in self.opts.get_fields(include_hidden=True) if f.auto_created and (not f.concrete))
        for related_object in related_objects:
            related_model = related_object.related_model
            remote_field = related_object.field.remote_field
            if any((issubclass(model, related_model) for model in registered_models)) and hasattr(remote_field, 'get_related_field') and (remote_field.get_related_field() == field):
                return True
        return False

    def has_add_permission(self, request):
        """
        Return True if the given request has permission to add an object.
        Can be overridden by the user in subclasses.
        """
        opts = self.opts
        codename = get_permission_codename('add', opts)
        return request.user.has_perm('%s.%s' % (opts.app_label, codename))

    def has_change_permission(self, request, obj=None):
        """
        Return True if the given request has permission to change the given
        Django model instance, the default implementation doesn't examine the
        `obj` parameter.

        Can be overridden by the user in subclasses. In such case it should
        return True if the given request has permission to change the `obj`
        model instance. If `obj` is None, this should return True if the given
        request has permission to change *any* object of the given type.
        """
        opts = self.opts
        codename = get_permission_codename('change', opts)
        return request.user.has_perm('%s.%s' % (opts.app_label, codename))

    def has_delete_permission(self, request, obj=None):
        """
        Return True if the given request has permission to delete the given
        Django model instance, the default implementation doesn't examine the
        `obj` parameter.

        Can be overridden by the user in subclasses. In such case it should
        return True if the given request has permission to delete the `obj`
        model instance. If `obj` is None, this should return True if the given
        request has permission to delete *any* object of the given type.
        """
        opts = self.opts
        codename = get_permission_codename('delete', opts)
        return request.user.has_perm('%s.%s' % (opts.app_label, codename))

    def has_view_permission(self, request, obj=None):
        """
        Return True if the given request has permission to view the given
        Django model instance. The default implementation doesn't examine the
        `obj` parameter.

        If overridden by the user in subclasses, it should return True if the
        given request has permission to view the `obj` model instance. If `obj`
        is None, it should return True if the request has permission to view
        any object of the given type.
        """
        opts = self.opts
        codename_view = get_permission_codename('view', opts)
        codename_change = get_permission_codename('change', opts)
        return request.user.has_perm('%s.%s' % (opts.app_label, codename_view)) or request.user.has_perm('%s.%s' % (opts.app_label, codename_change))

    def has_view_or_change_permission(self, request, obj=None):
        return self.has_view_permission(request, obj) or self.has_change_permission(request, obj)

    def has_module_permission(self, request):
        """
        Return True if the given request has any permission in the given
        app label.

        Can be overridden by the user in subclasses. In such case it should
        return True if the given request has permission to view the module on
        the admin index page and access the module's index page. Overriding it
        does not restrict access to the add, change or delete views. Use
        `ModelAdmin.has_(add|change|delete)_permission` for that.
        """
        return request.user.has_module_perms(self.opts.app_label)