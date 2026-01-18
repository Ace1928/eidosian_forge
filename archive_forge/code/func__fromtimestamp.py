import time as _time
import math as _math
import sys
from operator import index as _index
@classmethod
def _fromtimestamp(cls, t, utc, tz):
    """Construct a datetime from a POSIX timestamp (like time.time()).

        A timezone info object may be passed in as well.
        """
    frac, t = _math.modf(t)
    us = round(frac * 1000000.0)
    if us >= 1000000:
        t += 1
        us -= 1000000
    elif us < 0:
        t -= 1
        us += 1000000
    converter = _time.gmtime if utc else _time.localtime
    y, m, d, hh, mm, ss, weekday, jday, dst = converter(t)
    ss = min(ss, 59)
    result = cls(y, m, d, hh, mm, ss, us, tz)
    if tz is None and (not utc):
        max_fold_seconds = 24 * 3600
        if t < max_fold_seconds and sys.platform.startswith('win'):
            return result
        y, m, d, hh, mm, ss = converter(t - max_fold_seconds)[:6]
        probe1 = cls(y, m, d, hh, mm, ss, us, tz)
        trans = result - probe1 - timedelta(0, max_fold_seconds)
        if trans.days < 0:
            y, m, d, hh, mm, ss = converter(t + trans // timedelta(0, 1))[:6]
            probe2 = cls(y, m, d, hh, mm, ss, us, tz)
            if probe2 == result:
                result._fold = 1
    elif tz is not None:
        result = tz.fromutc(result)
    return result