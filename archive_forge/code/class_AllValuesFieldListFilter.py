import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class AllValuesFieldListFilter(FieldListFilter):

    def __init__(self, field, request, params, model, model_admin, field_path):
        self.lookup_kwarg = field_path
        self.lookup_kwarg_isnull = '%s__isnull' % field_path
        self.lookup_val = params.get(self.lookup_kwarg)
        self.lookup_val_isnull = get_last_value_from_parameters(params, self.lookup_kwarg_isnull)
        self.empty_value_display = model_admin.get_empty_value_display()
        parent_model, reverse_path = reverse_field_path(model, field_path)
        if model == parent_model:
            queryset = model_admin.get_queryset(request)
        else:
            queryset = parent_model._default_manager.all()
        self.lookup_choices = queryset.distinct().order_by(field.name).values_list(field.name, flat=True)
        super().__init__(field, request, params, model, model_admin, field_path)

    def expected_parameters(self):
        return [self.lookup_kwarg, self.lookup_kwarg_isnull]

    def get_facet_counts(self, pk_attname, filtered_qs):
        return {f'{i}__c': models.Count(pk_attname, filter=models.Q((self.lookup_kwarg, value) if value is not None else (self.lookup_kwarg_isnull, True))) for i, value in enumerate(self.lookup_choices)}

    def choices(self, changelist):
        add_facets = changelist.add_facets
        facet_counts = self.get_facet_queryset(changelist) if add_facets else None
        yield {'selected': self.lookup_val is None and self.lookup_val_isnull is None, 'query_string': changelist.get_query_string(remove=[self.lookup_kwarg, self.lookup_kwarg_isnull]), 'display': _('All')}
        include_none = False
        count = None
        empty_title = self.empty_value_display
        for i, val in enumerate(self.lookup_choices):
            if add_facets:
                count = facet_counts[f'{i}__c']
            if val is None:
                include_none = True
                empty_title = f'{empty_title} ({count})' if add_facets else empty_title
                continue
            val = str(val)
            yield {'selected': self.lookup_val is not None and val in self.lookup_val, 'query_string': changelist.get_query_string({self.lookup_kwarg: val}, [self.lookup_kwarg_isnull]), 'display': f'{val} ({count})' if add_facets else val}
        if include_none:
            yield {'selected': bool(self.lookup_val_isnull), 'query_string': changelist.get_query_string({self.lookup_kwarg_isnull: 'True'}, [self.lookup_kwarg]), 'display': empty_title}