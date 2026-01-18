from datetime import datetime
from django.conf import settings
from django.db.models.expressions import Func
from django.db.models.fields import (
from django.db.models.lookups import (
from django.utils import timezone
class ExtractIsoWeekDay(Extract):
    """Return Monday=1 through Sunday=7, based on ISO-8601."""
    lookup_name = 'iso_week_day'