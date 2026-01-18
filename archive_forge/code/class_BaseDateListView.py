import datetime
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import models
from django.http import Http404
from django.utils import timezone
from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from django.views.generic.base import View
from django.views.generic.detail import (
from django.views.generic.list import (
class BaseDateListView(MultipleObjectMixin, DateMixin, View):
    """Abstract base class for date-based views displaying a list of objects."""
    allow_empty = False
    date_list_period = 'year'

    def get(self, request, *args, **kwargs):
        self.date_list, self.object_list, extra_context = self.get_dated_items()
        context = self.get_context_data(object_list=self.object_list, date_list=self.date_list, **extra_context)
        return self.render_to_response(context)

    def get_dated_items(self):
        """Obtain the list of dates and items."""
        raise NotImplementedError('A DateView must provide an implementation of get_dated_items()')

    def get_ordering(self):
        """
        Return the field or fields to use for ordering the queryset; use the
        date field by default.
        """
        return '-%s' % self.get_date_field() if self.ordering is None else self.ordering

    def get_dated_queryset(self, **lookup):
        """
        Get a queryset properly filtered according to `allow_future` and any
        extra lookup kwargs.
        """
        qs = self.get_queryset().filter(**lookup)
        date_field = self.get_date_field()
        allow_future = self.get_allow_future()
        allow_empty = self.get_allow_empty()
        paginate_by = self.get_paginate_by(qs)
        if not allow_future:
            now = timezone.now() if self.uses_datetime_field else timezone_today()
            qs = qs.filter(**{'%s__lte' % date_field: now})
        if not allow_empty:
            is_empty = not qs if paginate_by is None else not qs.exists()
            if is_empty:
                raise Http404(_('No %(verbose_name_plural)s available') % {'verbose_name_plural': qs.model._meta.verbose_name_plural})
        return qs

    def get_date_list_period(self):
        """
        Get the aggregation period for the list of dates: 'year', 'month', or
        'day'.
        """
        return self.date_list_period

    def get_date_list(self, queryset, date_type=None, ordering='ASC'):
        """
        Get a date list by calling `queryset.dates/datetimes()`, checking
        along the way for empty lists that aren't allowed.
        """
        date_field = self.get_date_field()
        allow_empty = self.get_allow_empty()
        if date_type is None:
            date_type = self.get_date_list_period()
        if self.uses_datetime_field:
            date_list = queryset.datetimes(date_field, date_type, ordering)
        else:
            date_list = queryset.dates(date_field, date_type, ordering)
        if date_list is not None and (not date_list) and (not allow_empty):
            raise Http404(_('No %(verbose_name_plural)s available') % {'verbose_name_plural': queryset.model._meta.verbose_name_plural})
        return date_list