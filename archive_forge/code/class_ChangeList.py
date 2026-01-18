import warnings
from datetime import datetime, timedelta
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import FieldListFilter
from django.contrib.admin.exceptions import (
from django.contrib.admin.options import (
from django.contrib.admin.utils import (
from django.core.exceptions import (
from django.core.paginator import InvalidPage
from django.db.models import F, Field, ManyToOneRel, OrderBy
from django.db.models.expressions import Combinable
from django.urls import reverse
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.http import urlencode
from django.utils.inspect import func_supports_parameter
from django.utils.timezone import make_aware
from django.utils.translation import gettext
class ChangeList:
    search_form_class = ChangeListSearchForm

    def __init__(self, request, model, list_display, list_display_links, list_filter, date_hierarchy, search_fields, list_select_related, list_per_page, list_max_show_all, list_editable, model_admin, sortable_by, search_help_text):
        self.model = model
        self.opts = model._meta
        self.lookup_opts = self.opts
        self.root_queryset = model_admin.get_queryset(request)
        self.list_display = list_display
        self.list_display_links = list_display_links
        self.list_filter = list_filter
        self.has_filters = None
        self.has_active_filters = None
        self.clear_all_filters_qs = None
        self.date_hierarchy = date_hierarchy
        self.search_fields = search_fields
        self.list_select_related = list_select_related
        self.list_per_page = list_per_page
        self.list_max_show_all = list_max_show_all
        self.model_admin = model_admin
        self.preserved_filters = model_admin.get_preserved_filters(request)
        self.sortable_by = sortable_by
        self.search_help_text = search_help_text
        _search_form = self.search_form_class(request.GET)
        if not _search_form.is_valid():
            for error in _search_form.errors.values():
                messages.error(request, ', '.join(error))
        self.query = _search_form.cleaned_data.get(SEARCH_VAR) or ''
        try:
            self.page_num = int(request.GET.get(PAGE_VAR, 1))
        except ValueError:
            self.page_num = 1
        self.show_all = ALL_VAR in request.GET
        self.is_popup = IS_POPUP_VAR in request.GET
        self.add_facets = model_admin.show_facets is ShowFacets.ALWAYS or (model_admin.show_facets is ShowFacets.ALLOW and IS_FACETS_VAR in request.GET)
        self.is_facets_optional = model_admin.show_facets is ShowFacets.ALLOW
        to_field = request.GET.get(TO_FIELD_VAR)
        if to_field and (not model_admin.to_field_allowed(request, to_field)):
            raise DisallowedModelAdminToField('The field %s cannot be referenced.' % to_field)
        self.to_field = to_field
        self.params = dict(request.GET.items())
        self.filter_params = dict(request.GET.lists())
        if PAGE_VAR in self.params:
            del self.params[PAGE_VAR]
            del self.filter_params[PAGE_VAR]
        if ERROR_FLAG in self.params:
            del self.params[ERROR_FLAG]
            del self.filter_params[ERROR_FLAG]
        self.remove_facet_link = self.get_query_string(remove=[IS_FACETS_VAR])
        self.add_facet_link = self.get_query_string({IS_FACETS_VAR: True})
        if self.is_popup:
            self.list_editable = ()
        else:
            self.list_editable = list_editable
        self.queryset = self.get_queryset(request)
        self.get_results(request)
        if self.is_popup:
            title = gettext('Select %s')
        elif self.model_admin.has_change_permission(request):
            title = gettext('Select %s to change')
        else:
            title = gettext('Select %s to view')
        self.title = title % self.opts.verbose_name
        self.pk_attname = self.lookup_opts.pk.attname

    def __repr__(self):
        return '<%s: model=%s model_admin=%s>' % (self.__class__.__qualname__, self.model.__qualname__, self.model_admin.__class__.__qualname__)

    def get_filters_params(self, params=None):
        """
        Return all params except IGNORED_PARAMS.
        """
        params = params or self.filter_params
        lookup_params = params.copy()
        for ignored in IGNORED_PARAMS:
            if ignored in lookup_params:
                del lookup_params[ignored]
        return lookup_params

    def get_filters(self, request):
        lookup_params = self.get_filters_params()
        may_have_duplicates = False
        has_active_filters = False
        supports_request = func_supports_parameter(self.model_admin.lookup_allowed, 'request')
        if not supports_request:
            warnings.warn(f'`request` must be added to the signature of {self.model_admin.__class__.__qualname__}.lookup_allowed().', RemovedInDjango60Warning)
        for key, value_list in lookup_params.items():
            for value in value_list:
                params = (key, value, request) if supports_request else (key, value)
                if not self.model_admin.lookup_allowed(*params):
                    raise DisallowedModelAdminLookup(f'Filtering by {key} not allowed')
        filter_specs = []
        for list_filter in self.list_filter:
            lookup_params_count = len(lookup_params)
            if callable(list_filter):
                spec = list_filter(request, lookup_params, self.model, self.model_admin)
            else:
                field_path = None
                if isinstance(list_filter, (tuple, list)):
                    field, field_list_filter_class = list_filter
                else:
                    field, field_list_filter_class = (list_filter, FieldListFilter.create)
                if not isinstance(field, Field):
                    field_path = field
                    field = get_fields_from_path(self.model, field_path)[-1]
                spec = field_list_filter_class(field, request, lookup_params, self.model, self.model_admin, field_path=field_path)
                if lookup_params_count > len(lookup_params):
                    may_have_duplicates |= lookup_spawns_duplicates(self.lookup_opts, field_path)
            if spec and spec.has_output():
                filter_specs.append(spec)
                if lookup_params_count > len(lookup_params):
                    has_active_filters = True
        if self.date_hierarchy:
            year = lookup_params.pop('%s__year' % self.date_hierarchy, None)
            if year is not None:
                month = lookup_params.pop('%s__month' % self.date_hierarchy, None)
                day = lookup_params.pop('%s__day' % self.date_hierarchy, None)
                try:
                    from_date = datetime(int(year[-1]), int(month[-1] if month is not None else 1), int(day[-1] if day is not None else 1))
                except ValueError as e:
                    raise IncorrectLookupParameters(e) from e
                if day:
                    to_date = from_date + timedelta(days=1)
                elif month:
                    to_date = (from_date + timedelta(days=32)).replace(day=1)
                else:
                    to_date = from_date.replace(year=from_date.year + 1)
                if settings.USE_TZ:
                    from_date = make_aware(from_date)
                    to_date = make_aware(to_date)
                lookup_params.update({'%s__gte' % self.date_hierarchy: [from_date], '%s__lt' % self.date_hierarchy: [to_date]})
        try:
            for key, value in lookup_params.items():
                lookup_params[key] = prepare_lookup_value(key, value)
                may_have_duplicates |= lookup_spawns_duplicates(self.lookup_opts, key)
            return (filter_specs, bool(filter_specs), lookup_params, may_have_duplicates, has_active_filters)
        except FieldDoesNotExist as e:
            raise IncorrectLookupParameters(e) from e

    def get_query_string(self, new_params=None, remove=None):
        if new_params is None:
            new_params = {}
        if remove is None:
            remove = []
        p = self.filter_params.copy()
        for r in remove:
            for k in list(p):
                if k.startswith(r):
                    del p[k]
        for k, v in new_params.items():
            if v is None:
                if k in p:
                    del p[k]
            else:
                p[k] = v
        return '?%s' % urlencode(sorted(p.items()), doseq=True)

    def get_results(self, request):
        paginator = self.model_admin.get_paginator(request, self.queryset, self.list_per_page)
        result_count = paginator.count
        if self.model_admin.show_full_result_count:
            full_result_count = self.root_queryset.count()
        else:
            full_result_count = None
        can_show_all = result_count <= self.list_max_show_all
        multi_page = result_count > self.list_per_page
        if self.show_all and can_show_all or not multi_page:
            result_list = self.queryset._clone()
        else:
            try:
                result_list = paginator.page(self.page_num).object_list
            except InvalidPage:
                raise IncorrectLookupParameters
        self.result_count = result_count
        self.show_full_result_count = self.model_admin.show_full_result_count
        self.show_admin_actions = not self.show_full_result_count or bool(full_result_count)
        self.full_result_count = full_result_count
        self.result_list = result_list
        self.can_show_all = can_show_all
        self.multi_page = multi_page
        self.paginator = paginator

    def _get_default_ordering(self):
        ordering = []
        if self.model_admin.ordering:
            ordering = self.model_admin.ordering
        elif self.lookup_opts.ordering:
            ordering = self.lookup_opts.ordering
        return ordering

    def get_ordering_field(self, field_name):
        """
        Return the proper model field name corresponding to the given
        field_name to use for ordering. field_name may either be the name of a
        proper model field or the name of a method (on the admin or model) or a
        callable with the 'admin_order_field' attribute. Return None if no
        proper model field name can be matched.
        """
        try:
            field = self.lookup_opts.get_field(field_name)
            return field.name
        except FieldDoesNotExist:
            if callable(field_name):
                attr = field_name
            elif hasattr(self.model_admin, field_name):
                attr = getattr(self.model_admin, field_name)
            else:
                attr = getattr(self.model, field_name)
            if isinstance(attr, property) and hasattr(attr, 'fget'):
                attr = attr.fget
            return getattr(attr, 'admin_order_field', None)

    def get_ordering(self, request, queryset):
        """
        Return the list of ordering fields for the change list.
        First check the get_ordering() method in model admin, then check
        the object's default ordering. Then, any manually-specified ordering
        from the query string overrides anything. Finally, a deterministic
        order is guaranteed by calling _get_deterministic_ordering() with the
        constructed ordering.
        """
        params = self.params
        ordering = list(self.model_admin.get_ordering(request) or self._get_default_ordering())
        if ORDER_VAR in params:
            ordering = []
            order_params = params[ORDER_VAR].split('.')
            for p in order_params:
                try:
                    none, pfx, idx = p.rpartition('-')
                    field_name = self.list_display[int(idx)]
                    order_field = self.get_ordering_field(field_name)
                    if not order_field:
                        continue
                    if isinstance(order_field, OrderBy):
                        if pfx == '-':
                            order_field = order_field.copy()
                            order_field.reverse_ordering()
                        ordering.append(order_field)
                    elif hasattr(order_field, 'resolve_expression'):
                        ordering.append(order_field.desc() if pfx == '-' else order_field.asc())
                    elif pfx == '-' and order_field.startswith(pfx):
                        ordering.append(order_field.removeprefix(pfx))
                    else:
                        ordering.append(pfx + order_field)
                except (IndexError, ValueError):
                    continue
        ordering.extend(queryset.query.order_by)
        return self._get_deterministic_ordering(ordering)

    def _get_deterministic_ordering(self, ordering):
        """
        Ensure a deterministic order across all database backends. Search for a
        single field or unique together set of fields providing a total
        ordering. If these are missing, augment the ordering with a descendant
        primary key.
        """
        ordering = list(ordering)
        ordering_fields = set()
        total_ordering_fields = {'pk'} | {field.attname for field in self.lookup_opts.fields if field.unique and (not field.null)}
        for part in ordering:
            field_name = None
            if isinstance(part, str):
                field_name = part.lstrip('-')
            elif isinstance(part, F):
                field_name = part.name
            elif isinstance(part, OrderBy) and isinstance(part.expression, F):
                field_name = part.expression.name
            if field_name:
                try:
                    field = self.lookup_opts.get_field(field_name)
                except FieldDoesNotExist:
                    continue
                if field.remote_field and field_name == field.name:
                    continue
                if field.attname in total_ordering_fields:
                    break
                ordering_fields.add(field.attname)
        else:
            constraint_field_names = (*self.lookup_opts.unique_together, *(constraint.fields for constraint in self.lookup_opts.total_unique_constraints))
            for field_names in constraint_field_names:
                fields = [self.lookup_opts.get_field(field_name) for field_name in field_names]
                if any((field.null for field in fields)):
                    continue
                if ordering_fields.issuperset((field.attname for field in fields)):
                    break
            else:
                ordering.append('-pk')
        return ordering

    def get_ordering_field_columns(self):
        """
        Return a dictionary of ordering field column numbers and asc/desc.
        """
        ordering = self._get_default_ordering()
        ordering_fields = {}
        if ORDER_VAR not in self.params:
            for field in ordering:
                if isinstance(field, (Combinable, OrderBy)):
                    if not isinstance(field, OrderBy):
                        field = field.asc()
                    if isinstance(field.expression, F):
                        order_type = 'desc' if field.descending else 'asc'
                        field = field.expression.name
                    else:
                        continue
                elif field.startswith('-'):
                    field = field.removeprefix('-')
                    order_type = 'desc'
                else:
                    order_type = 'asc'
                for index, attr in enumerate(self.list_display):
                    if self.get_ordering_field(attr) == field:
                        ordering_fields[index] = order_type
                        break
        else:
            for p in self.params[ORDER_VAR].split('.'):
                none, pfx, idx = p.rpartition('-')
                try:
                    idx = int(idx)
                except ValueError:
                    continue
                ordering_fields[idx] = 'desc' if pfx == '-' else 'asc'
        return ordering_fields

    def get_queryset(self, request, exclude_parameters=None):
        self.filter_specs, self.has_filters, remaining_lookup_params, filters_may_have_duplicates, self.has_active_filters = self.get_filters(request)
        qs = self.root_queryset
        for filter_spec in self.filter_specs:
            if exclude_parameters is None or filter_spec.expected_parameters() != exclude_parameters:
                new_qs = filter_spec.queryset(request, qs)
                if new_qs is not None:
                    qs = new_qs
        try:
            q_object = build_q_object_from_lookup_parameters(remaining_lookup_params)
            qs = qs.filter(q_object)
        except (SuspiciousOperation, ImproperlyConfigured):
            raise
        except Exception as e:
            raise IncorrectLookupParameters(e)
        if not qs.query.select_related:
            qs = self.apply_select_related(qs)
        ordering = self.get_ordering(request, qs)
        qs = qs.order_by(*ordering)
        qs, search_may_have_duplicates = self.model_admin.get_search_results(request, qs, self.query)
        self.clear_all_filters_qs = self.get_query_string(new_params=remaining_lookup_params, remove=self.get_filters_params())
        if filters_may_have_duplicates | search_may_have_duplicates:
            return qs.distinct()
        else:
            return qs

    def apply_select_related(self, qs):
        if self.list_select_related is True:
            return qs.select_related()
        if self.list_select_related is False:
            if self.has_related_field_in_list_display():
                return qs.select_related()
        if self.list_select_related:
            return qs.select_related(*self.list_select_related)
        return qs

    def has_related_field_in_list_display(self):
        for field_name in self.list_display:
            try:
                field = self.lookup_opts.get_field(field_name)
            except FieldDoesNotExist:
                pass
            else:
                if isinstance(field.remote_field, ManyToOneRel):
                    if field_name != field.get_attname():
                        return True
        return False

    def url_for_result(self, result):
        pk = getattr(result, self.pk_attname)
        return reverse('admin:%s_%s_change' % (self.opts.app_label, self.opts.model_name), args=(quote(pk),), current_app=self.model_admin.admin_site.name)