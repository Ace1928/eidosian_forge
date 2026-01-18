import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
@staticmethod
def _iso_year_start(isoyear):
    fourth_jan = datetime.date(isoyear, 1, 4)
    delta = datetime.timedelta(days=fourth_jan.isoweekday() - 1)
    return fourth_jan - delta