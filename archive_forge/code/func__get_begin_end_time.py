import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
@staticmethod
def _get_begin_end_time():
    current = timeutils.utcnow()
    end = datetime.datetime(day=1, month=current.month, year=current.year)
    year = end.year
    if current.month == 1:
        year -= 1
        month = 12
    else:
        month = current.month - 1
    begin = datetime.datetime(day=1, month=month, year=year)
    return (begin, end)