from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def _parse_datespec(spec):
    import datetime
    today = datetime.datetime.fromordinal(datetime.date.today().toordinal())
    if spec.lower() == 'yesterday':
        return today - datetime.timedelta(days=1)
    elif spec.lower() == 'today':
        return today
    elif spec.lower() == 'tomorrow':
        return today + datetime.timedelta(days=1)
    else:
        m = _date_regex.match(spec)
        if not m or (not m.group('date') and (not m.group('time'))):
            raise ValueError
        if m.group('date'):
            year = int(m.group('year'))
            month = int(m.group('month'))
            day = int(m.group('day'))
        else:
            year = today.year
            month = today.month
            day = today.day
        if m.group('time'):
            hour = int(m.group('hour'))
            minute = int(m.group('minute'))
            if m.group('second'):
                second = int(m.group('second'))
            else:
                second = 0
        else:
            hour, minute, second = (0, 0, 0)
        return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)