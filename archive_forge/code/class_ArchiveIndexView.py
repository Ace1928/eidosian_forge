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
class ArchiveIndexView(MultipleObjectTemplateResponseMixin, BaseArchiveIndexView):
    """Top-level archive of date-based items."""
    template_name_suffix = '_archive'