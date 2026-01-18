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
class MonthMixin:
    """Mixin for views manipulating month-based data."""
    month_format = '%b'
    month = None

    def get_month_format(self):
        """
        Get a month format string in strptime syntax to be used to parse the
        month from url variables.
        """
        return self.month_format

    def get_month(self):
        """Return the month for which this view should display data."""
        month = self.month
        if month is None:
            try:
                month = self.kwargs['month']
            except KeyError:
                try:
                    month = self.request.GET['month']
                except KeyError:
                    raise Http404(_('No month specified'))
        return month

    def get_next_month(self, date):
        """Get the next valid month."""
        return _get_next_prev(self, date, is_previous=False, period='month')

    def get_previous_month(self, date):
        """Get the previous valid month."""
        return _get_next_prev(self, date, is_previous=True, period='month')

    def _get_next_month(self, date):
        """
        Return the start date of the next interval.

        The interval is defined by start date <= item date < next start date.
        """
        if date.month == 12:
            try:
                return date.replace(year=date.year + 1, month=1, day=1)
            except ValueError:
                raise Http404(_('Date out of range'))
        else:
            return date.replace(month=date.month + 1, day=1)

    def _get_current_month(self, date):
        """Return the start date of the previous interval."""
        return date.replace(day=1)