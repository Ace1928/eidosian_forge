from suds import UnicodeMixin
import datetime
import re
import time
@staticmethod
def __parse(value):
    """
        Parse the string date.

        Supports the subset of ISO8601 used by xsd:time, but is lenient with
        what is accepted, handling most reasonable syntax.

        Subsecond information is rounded to microseconds due to a restriction
        in the python datetime.time implementation.

        @param value: A time string.
        @type value: str
        @return: A time object.
        @rtype: B{datetime}.I{time}

        """
    match_result = _RE_TIME.match(value)
    if match_result is None:
        raise ValueError("date data has invalid format '%s'" % (value,))
    time, round_up = _time_from_match(match_result)
    tzinfo = _tzinfo_from_match(match_result)
    if round_up:
        time = _bump_up_time_by_microsecond(time)
    return time.replace(tzinfo=tzinfo)