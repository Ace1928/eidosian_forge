import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class RelatedFieldListFilter(FieldListFilter):

    def __init__(self, field, request, params, model, model_admin, field_path):
        other_model = get_model_from_relation(field)
        self.lookup_kwarg = '%s__%s__exact' % (field_path, field.target_field.name)
        self.lookup_kwarg_isnull = '%s__isnull' % field_path
        self.lookup_val = params.get(self.lookup_kwarg)
        self.lookup_val_isnull = get_last_value_from_parameters(params, self.lookup_kwarg_isnull)
        super().__init__(field, request, params, model, model_admin, field_path)
        self.lookup_choices = self.field_choices(field, request, model_admin)
        if hasattr(field, 'verbose_name'):
            self.lookup_title = field.verbose_name
        else:
            self.lookup_title = other_model._meta.verbose_name
        self.title = self.lookup_title
        self.empty_value_display = model_admin.get_empty_value_display()

    @property
    def include_empty_choice(self):
        """
        Return True if a "(None)" choice should be included, which filters
        out everything except empty relationships.
        """
        return self.field.null or (self.field.is_relation and self.field.many_to_many)

    def has_output(self):
        if self.include_empty_choice:
            extra = 1
        else:
            extra = 0
        return len(self.lookup_choices) + extra > 1

    def expected_parameters(self):
        return [self.lookup_kwarg, self.lookup_kwarg_isnull]

    def field_admin_ordering(self, field, request, model_admin):
        """
        Return the model admin's ordering for related field, if provided.
        """
        try:
            related_admin = model_admin.admin_site.get_model_admin(field.remote_field.model)
        except NotRegistered:
            return ()
        else:
            return related_admin.get_ordering(request)

    def field_choices(self, field, request, model_admin):
        ordering = self.field_admin_ordering(field, request, model_admin)
        return field.get_choices(include_blank=False, ordering=ordering)

    def get_facet_counts(self, pk_attname, filtered_qs):
        counts = {f'{pk_val}__c': models.Count(pk_attname, filter=models.Q(**{self.lookup_kwarg: pk_val})) for pk_val, _ in self.lookup_choices}
        if self.include_empty_choice:
            counts['__c'] = models.Count(pk_attname, filter=models.Q(**{self.lookup_kwarg_isnull: True}))
        return counts

    def choices(self, changelist):
        add_facets = changelist.add_facets
        facet_counts = self.get_facet_queryset(changelist) if add_facets else None
        yield {'selected': self.lookup_val is None and (not self.lookup_val_isnull), 'query_string': changelist.get_query_string(remove=[self.lookup_kwarg, self.lookup_kwarg_isnull]), 'display': _('All')}
        count = None
        for pk_val, val in self.lookup_choices:
            if add_facets:
                count = facet_counts[f'{pk_val}__c']
                val = f'{val} ({count})'
            yield {'selected': self.lookup_val is not None and str(pk_val) in self.lookup_val, 'query_string': changelist.get_query_string({self.lookup_kwarg: pk_val}, [self.lookup_kwarg_isnull]), 'display': val}
        empty_title = self.empty_value_display
        if self.include_empty_choice:
            if add_facets:
                count = facet_counts['__c']
                empty_title = f'{empty_title} ({count})'
            yield {'selected': bool(self.lookup_val_isnull), 'query_string': changelist.get_query_string({self.lookup_kwarg_isnull: 'True'}, [self.lookup_kwarg]), 'display': empty_title}