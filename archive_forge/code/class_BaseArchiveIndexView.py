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
class BaseArchiveIndexView(BaseDateListView):
    """
    Base class for archives of date-based items. Requires a response mixin.
    """
    context_object_name = 'latest'

    def get_dated_items(self):
        """Return (date_list, items, extra_context) for this request."""
        qs = self.get_dated_queryset()
        date_list = self.get_date_list(qs, ordering='DESC')
        if not date_list:
            qs = qs.none()
        return (date_list, qs, {})