import re
from .w3dtf import _parse_date_w3dtf
def _parse_date_hungarian(date_string):
    """Parse a string according to a Hungarian 8-bit date format."""
    m = _hungarian_date_format_re.match(date_string)
    if not m or m.group(2) not in _hungarian_months:
        return None
    month = _hungarian_months[m.group(2)]
    day = m.group(3)
    if len(day) == 1:
        day = '0' + day
    hour = m.group(4)
    if len(hour) == 1:
        hour = '0' + hour
    w3dtfdate = '%(year)s-%(month)s-%(day)sT%(hour)s:%(minute)s%(zonediff)s' % {'year': m.group(1), 'month': month, 'day': day, 'hour': hour, 'minute': m.group(5), 'zonediff': m.group(6)}
    return _parse_date_w3dtf(w3dtfdate)