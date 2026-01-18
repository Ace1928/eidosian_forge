import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class DateFieldListFilter(FieldListFilter):

    def __init__(self, field, request, params, model, model_admin, field_path):
        self.field_generic = '%s__' % field_path
        self.date_params = {k: v[-1] for k, v in params.items() if k.startswith(self.field_generic)}
        now = timezone.now()
        if timezone.is_aware(now):
            now = timezone.localtime(now)
        if isinstance(field, models.DateTimeField):
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            today = now.date()
        tomorrow = today + datetime.timedelta(days=1)
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        next_year = today.replace(year=today.year + 1, month=1, day=1)
        self.lookup_kwarg_since = '%s__gte' % field_path
        self.lookup_kwarg_until = '%s__lt' % field_path
        self.links = ((_('Any date'), {}), (_('Today'), {self.lookup_kwarg_since: today, self.lookup_kwarg_until: tomorrow}), (_('Past 7 days'), {self.lookup_kwarg_since: today - datetime.timedelta(days=7), self.lookup_kwarg_until: tomorrow}), (_('This month'), {self.lookup_kwarg_since: today.replace(day=1), self.lookup_kwarg_until: next_month}), (_('This year'), {self.lookup_kwarg_since: today.replace(month=1, day=1), self.lookup_kwarg_until: next_year}))
        if field.null:
            self.lookup_kwarg_isnull = '%s__isnull' % field_path
            self.links += ((_('No date'), {self.field_generic + 'isnull': True}), (_('Has date'), {self.field_generic + 'isnull': False}))
        super().__init__(field, request, params, model, model_admin, field_path)

    def expected_parameters(self):
        params = [self.lookup_kwarg_since, self.lookup_kwarg_until]
        if self.field.null:
            params.append(self.lookup_kwarg_isnull)
        return params

    def get_facet_counts(self, pk_attname, filtered_qs):
        return {f'{i}__c': models.Count(pk_attname, filter=models.Q(**param_dict)) for i, (_, param_dict) in enumerate(self.links)}

    def choices(self, changelist):
        add_facets = changelist.add_facets
        facet_counts = self.get_facet_queryset(changelist) if add_facets else None
        for i, (title, param_dict) in enumerate(self.links):
            param_dict_str = {key: str(value) for key, value in param_dict.items()}
            if add_facets:
                count = facet_counts[f'{i}__c']
                title = f'{title} ({count})'
            yield {'selected': self.date_params == param_dict_str, 'query_string': changelist.get_query_string(param_dict_str, [self.field_generic]), 'display': title}