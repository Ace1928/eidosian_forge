import datetime
import decimal
import uuid
from functools import lru_cache
from itertools import chain
from django.conf import settings
from django.core.exceptions import FieldError
from django.db import DatabaseError, NotSupportedError, models
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.constants import OnConflict
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.functional import cached_property
def _convert_tznames_to_sql(self, tzname):
    if tzname and settings.USE_TZ:
        return (tzname, self.connection.timezone_name)
    return (None, None)