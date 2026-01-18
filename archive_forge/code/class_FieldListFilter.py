import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class FieldListFilter(FacetsMixin, ListFilter):
    _field_list_filters = []
    _take_priority_index = 0
    list_separator = ','

    def __init__(self, field, request, params, model, model_admin, field_path):
        self.field = field
        self.field_path = field_path
        self.title = getattr(field, 'verbose_name', field_path)
        super().__init__(request, params, model, model_admin)
        for p in self.expected_parameters():
            if p in params:
                value = params.pop(p)
                self.used_parameters[p] = prepare_lookup_value(p, value, self.list_separator)

    def has_output(self):
        return True

    def queryset(self, request, queryset):
        try:
            q_object = build_q_object_from_lookup_parameters(self.used_parameters)
            return queryset.filter(q_object)
        except (ValueError, ValidationError) as e:
            raise IncorrectLookupParameters(e)

    @classmethod
    def register(cls, test, list_filter_class, take_priority=False):
        if take_priority:
            cls._field_list_filters.insert(cls._take_priority_index, (test, list_filter_class))
            cls._take_priority_index += 1
        else:
            cls._field_list_filters.append((test, list_filter_class))

    @classmethod
    def create(cls, field, request, params, model, model_admin, field_path):
        for test, list_filter_class in cls._field_list_filters:
            if test(field):
                return list_filter_class(field, request, params, model, model_admin, field_path=field_path)