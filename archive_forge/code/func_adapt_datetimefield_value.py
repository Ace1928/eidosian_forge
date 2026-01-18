import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def adapt_datetimefield_value(self, value):
    if value is None:
        return None
    if hasattr(value, 'resolve_expression'):
        return value
    if timezone.is_aware(value):
        if settings.USE_TZ:
            value = timezone.make_naive(value, self.connection.timezone)
        else:
            raise ValueError('MySQL backend does not support timezone-aware datetimes when USE_TZ is False.')
    return str(value)