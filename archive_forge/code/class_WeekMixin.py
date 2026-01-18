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
class WeekMixin:
    """Mixin for views manipulating week-based data."""
    week_format = '%U'
    week = None

    def get_week_format(self):
        """
        Get a week format string in strptime syntax to be used to parse the
        week from url variables.
        """
        return self.week_format

    def get_week(self):
        """Return the week for which this view should display data."""
        week = self.week
        if week is None:
            try:
                week = self.kwargs['week']
            except KeyError:
                try:
                    week = self.request.GET['week']
                except KeyError:
                    raise Http404(_('No week specified'))
        return week

    def get_next_week(self, date):
        """Get the next valid week."""
        return _get_next_prev(self, date, is_previous=False, period='week')

    def get_previous_week(self, date):
        """Get the previous valid week."""
        return _get_next_prev(self, date, is_previous=True, period='week')

    def _get_next_week(self, date):
        """
        Return the start date of the next interval.

        The interval is defined by start date <= item date < next start date.
        """
        try:
            return date + datetime.timedelta(days=7 - self._get_weekday(date))
        except OverflowError:
            raise Http404(_('Date out of range'))

    def _get_current_week(self, date):
        """Return the start date of the current interval."""
        return date - datetime.timedelta(self._get_weekday(date))

    def _get_weekday(self, date):
        """
        Return the weekday for a given date.

        The first day according to the week format is 0 and the last day is 6.
        """
        week_format = self.get_week_format()
        if week_format in {'%W', '%V'}:
            return date.weekday()
        elif week_format == '%U':
            return (date.weekday() + 1) % 7
        else:
            raise ValueError('unknown week format: %s' % week_format)