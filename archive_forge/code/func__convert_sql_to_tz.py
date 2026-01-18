import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def _convert_sql_to_tz(self, sql, params, tzname):
    if tzname and settings.USE_TZ and (self.connection.timezone_name != tzname):
        return (f'CONVERT_TZ({sql}, %s, %s)', (*params, self.connection.timezone_name, self._prepare_tzname_delta(tzname)))
    return (sql, params)