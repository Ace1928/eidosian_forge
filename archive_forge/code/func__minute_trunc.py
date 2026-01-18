import uuid
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
def _minute_trunc(ts):
    return ts.replace(second=0, microsecond=0)