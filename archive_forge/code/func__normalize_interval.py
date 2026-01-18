from calendar import timegm
from datetime import datetime, time, timedelta
from email.utils import parsedate_tz, mktime_tz
import re
import aniso8601
import pytz
def _normalize_interval(start, end, value):
    """Normalize datetime intervals.

    Given a pair of datetime.date or datetime.datetime objects,
    returns a 2-tuple of tz-aware UTC datetimes spanning the same interval.

    For datetime.date objects, the returned interval starts at 00:00:00.0
    on the first date and ends at 00:00:00.0 on the second.

    Naive datetimes are upgraded to UTC.

    Timezone-aware datetimes are normalized to the UTC tzdata.

    Params:
        - start: A date or datetime
        - end: A date or datetime
    """
    if not isinstance(start, datetime):
        start = datetime.combine(start, START_OF_DAY)
        end = datetime.combine(end, START_OF_DAY)
    if start.tzinfo is None:
        start = pytz.UTC.localize(start)
        end = pytz.UTC.localize(end)
    else:
        start = start.astimezone(pytz.UTC)
        end = end.astimezone(pytz.UTC)
    return (start, end)