import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class EmptyFieldListFilter(FieldListFilter):

    def __init__(self, field, request, params, model, model_admin, field_path):
        if not field.empty_strings_allowed and (not field.null):
            raise ImproperlyConfigured("The list filter '%s' cannot be used with field '%s' which doesn't allow empty strings and nulls." % (self.__class__.__name__, field.name))
        self.lookup_kwarg = '%s__isempty' % field_path
        self.lookup_val = get_last_value_from_parameters(params, self.lookup_kwarg)
        super().__init__(field, request, params, model, model_admin, field_path)

    def get_lookup_condition(self):
        lookup_conditions = []
        if self.field.empty_strings_allowed:
            lookup_conditions.append((self.field_path, ''))
        if self.field.null:
            lookup_conditions.append((f'{self.field_path}__isnull', True))
        return models.Q.create(lookup_conditions, connector=models.Q.OR)

    def queryset(self, request, queryset):
        if self.lookup_kwarg not in self.used_parameters:
            return queryset
        if self.lookup_val not in ('0', '1'):
            raise IncorrectLookupParameters
        lookup_condition = self.get_lookup_condition()
        if self.lookup_val == '1':
            return queryset.filter(lookup_condition)
        return queryset.exclude(lookup_condition)

    def expected_parameters(self):
        return [self.lookup_kwarg]

    def get_facet_counts(self, pk_attname, filtered_qs):
        lookup_condition = self.get_lookup_condition()
        return {'empty__c': models.Count(pk_attname, filter=lookup_condition), 'not_empty__c': models.Count(pk_attname, filter=~lookup_condition)}

    def choices(self, changelist):
        add_facets = changelist.add_facets
        facet_counts = self.get_facet_queryset(changelist) if add_facets else None
        for lookup, title, count_field in ((None, _('All'), None), ('1', _('Empty'), 'empty__c'), ('0', _('Not empty'), 'not_empty__c')):
            if add_facets:
                if count_field is not None:
                    count = facet_counts[count_field]
                    title = f'{title} ({count})'
            yield {'selected': self.lookup_val == lookup, 'query_string': changelist.get_query_string({self.lookup_kwarg: lookup}), 'display': title}