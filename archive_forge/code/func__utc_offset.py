from datetime import (
import time
import unittest
def _utc_offset(timestamp, use_system_timezone):
    """
    Return the UTC offset of `timestamp`. If `timestamp` does not have any `tzinfo`, use
    the timezone informations stored locally on the system.

    >>> if time.localtime().tm_isdst:
    ...     system_timezone = -time.altzone
    ... else:
    ...     system_timezone = -time.timezone
    >>> _utc_offset(datetime.now(), True) == system_timezone
    True
    >>> _utc_offset(datetime.now(), False)
    0
    """
    if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
        return _timedelta_to_seconds(timestamp.utcoffset())
    elif use_system_timezone:
        if timestamp.year < 1970:
            t = time.mktime(timestamp.replace(year=1972).timetuple())
        else:
            t = time.mktime(timestamp.timetuple())
        if time.localtime(t).tm_isdst:
            return -time.altzone
        else:
            return -time.timezone
    else:
        return 0