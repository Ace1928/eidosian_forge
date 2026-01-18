import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class ChoicesFieldListFilter(FieldListFilter):

    def __init__(self, field, request, params, model, model_admin, field_path):
        self.lookup_kwarg = '%s__exact' % field_path
        self.lookup_kwarg_isnull = '%s__isnull' % field_path
        self.lookup_val = params.get(self.lookup_kwarg)
        self.lookup_val_isnull = get_last_value_from_parameters(params, self.lookup_kwarg_isnull)
        super().__init__(field, request, params, model, model_admin, field_path)

    def expected_parameters(self):
        return [self.lookup_kwarg, self.lookup_kwarg_isnull]

    def get_facet_counts(self, pk_attname, filtered_qs):
        return {f'{i}__c': models.Count(pk_attname, filter=models.Q((self.lookup_kwarg, value) if value is not None else (self.lookup_kwarg_isnull, True))) for i, (value, _) in enumerate(self.field.flatchoices)}

    def choices(self, changelist):
        add_facets = changelist.add_facets
        facet_counts = self.get_facet_queryset(changelist) if add_facets else None
        yield {'selected': self.lookup_val is None, 'query_string': changelist.get_query_string(remove=[self.lookup_kwarg, self.lookup_kwarg_isnull]), 'display': _('All')}
        none_title = ''
        for i, (lookup, title) in enumerate(self.field.flatchoices):
            if add_facets:
                count = facet_counts[f'{i}__c']
                title = f'{title} ({count})'
            if lookup is None:
                none_title = title
                continue
            yield {'selected': self.lookup_val is not None and str(lookup) in self.lookup_val, 'query_string': changelist.get_query_string({self.lookup_kwarg: lookup}, [self.lookup_kwarg_isnull]), 'display': title}
        if none_title:
            yield {'selected': bool(self.lookup_val_isnull), 'query_string': changelist.get_query_string({self.lookup_kwarg_isnull: 'True'}, [self.lookup_kwarg]), 'display': none_title}