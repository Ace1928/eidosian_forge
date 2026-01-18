import time as _time
import math as _math
import sys
from operator import index as _index
def _parse_isoformat_date(dtstr):
    assert len(dtstr) in (7, 8, 10)
    year = int(dtstr[0:4])
    has_sep = dtstr[4] == '-'
    pos = 4 + has_sep
    if dtstr[pos:pos + 1] == 'W':
        pos += 1
        weekno = int(dtstr[pos:pos + 2])
        pos += 2
        dayno = 1
        if len(dtstr) > pos:
            if (dtstr[pos:pos + 1] == '-') != has_sep:
                raise ValueError('Inconsistent use of dash separator')
            pos += has_sep
            dayno = int(dtstr[pos:pos + 1])
        return list(_isoweek_to_gregorian(year, weekno, dayno))
    else:
        month = int(dtstr[pos:pos + 2])
        pos += 2
        if (dtstr[pos:pos + 1] == '-') != has_sep:
            raise ValueError('Inconsistent use of dash separator')
        pos += has_sep
        day = int(dtstr[pos:pos + 2])
        return [year, month, day]