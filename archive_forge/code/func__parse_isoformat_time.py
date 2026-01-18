import time as _time
import math as _math
import sys
from operator import index as _index
def _parse_isoformat_time(tstr):
    len_str = len(tstr)
    if len_str < 2:
        raise ValueError('Isoformat time too short')
    tz_pos = tstr.find('-') + 1 or tstr.find('+') + 1 or tstr.find('Z') + 1
    timestr = tstr[:tz_pos - 1] if tz_pos > 0 else tstr
    time_comps = _parse_hh_mm_ss_ff(timestr)
    tzi = None
    if tz_pos == len_str and tstr[-1] == 'Z':
        tzi = timezone.utc
    elif tz_pos > 0:
        tzstr = tstr[tz_pos:]
        if len(tzstr) in (0, 1, 3):
            raise ValueError('Malformed time zone string')
        tz_comps = _parse_hh_mm_ss_ff(tzstr)
        if all((x == 0 for x in tz_comps)):
            tzi = timezone.utc
        else:
            tzsign = -1 if tstr[tz_pos - 1] == '-' else 1
            td = timedelta(hours=tz_comps[0], minutes=tz_comps[1], seconds=tz_comps[2], microseconds=tz_comps[3])
            tzi = timezone(tzsign * td)
    time_comps.append(tzi)
    return time_comps