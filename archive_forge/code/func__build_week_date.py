import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
@staticmethod
def _build_week_date(isoyear, isoweek, isoday=None):
    if isoday is None:
        return PythonTimeBuilder._iso_year_start(isoyear) + datetime.timedelta(weeks=isoweek - 1)
    return PythonTimeBuilder._iso_year_start(isoyear) + datetime.timedelta(weeks=isoweek - 1, days=isoday - 1)