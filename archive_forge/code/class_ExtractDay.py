from datetime import datetime
from django.conf import settings
from django.db.models.expressions import Func
from django.db.models.fields import (
from django.db.models.lookups import (
from django.utils import timezone
class ExtractDay(Extract):
    lookup_name = 'day'