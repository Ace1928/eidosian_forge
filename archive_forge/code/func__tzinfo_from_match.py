from suds import UnicodeMixin
import datetime
import re
import time
def _tzinfo_from_match(match_object):
    """
    Create a timezone information object from a regular expression match.

    The regular expression match is expected to be from _RE_DATE, _RE_DATETIME
    or _RE_TIME.

    @param match_object: The regular expression match.
    @type match_object: B{re}.I{MatchObject}
    @return: A timezone information object.
    @rtype: B{datetime}.I{tzinfo}

    """
    tz_utc = match_object.group('tz_utc')
    if tz_utc:
        return UtcTimezone()
    tz_sign = match_object.group('tz_sign')
    if not tz_sign:
        return
    h = int(match_object.group('tz_hour') or 0)
    m = int(match_object.group('tz_minute') or 0)
    if h == 0 and m == 0:
        return UtcTimezone()
    if h >= 24:
        raise ValueError('timezone indicator too large')
    tz_delta = datetime.timedelta(hours=h, minutes=m)
    if tz_sign == '-':
        tz_delta *= -1
    return FixedOffsetTimezone(tz_delta)