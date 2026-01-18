import sys
import datetime
import os.path
from pytz.exceptions import AmbiguousTimeError
from pytz.exceptions import InvalidTimeError
from pytz.exceptions import NonExistentTimeError
from pytz.exceptions import UnknownTimeZoneError
from pytz.lazy import LazyDict, LazyList, LazySet  # noqa
from pytz.tzinfo import unpickler, BaseTzInfo
from pytz.tzfile import build_tzinfo
def FixedOffset(offset, _tzinfos={}):
    """return a fixed-offset timezone based off a number of minutes.

        >>> one = FixedOffset(-330)
        >>> one
        pytz.FixedOffset(-330)
        >>> str(one.utcoffset(datetime.datetime.now()))
        '-1 day, 18:30:00'
        >>> str(one.dst(datetime.datetime.now()))
        '0:00:00'

        >>> two = FixedOffset(1380)
        >>> two
        pytz.FixedOffset(1380)
        >>> str(two.utcoffset(datetime.datetime.now()))
        '23:00:00'
        >>> str(two.dst(datetime.datetime.now()))
        '0:00:00'

    The datetime.timedelta must be between the range of -1 and 1 day,
    non-inclusive.

        >>> FixedOffset(1440)
        Traceback (most recent call last):
        ...
        ValueError: ('absolute offset is too large', 1440)

        >>> FixedOffset(-1440)
        Traceback (most recent call last):
        ...
        ValueError: ('absolute offset is too large', -1440)

    An offset of 0 is special-cased to return UTC.

        >>> FixedOffset(0) is UTC
        True

    There should always be only one instance of a FixedOffset per timedelta.
    This should be true for multiple creation calls.

        >>> FixedOffset(-330) is one
        True
        >>> FixedOffset(1380) is two
        True

    It should also be true for pickling.

        >>> import pickle
        >>> pickle.loads(pickle.dumps(one)) is one
        True
        >>> pickle.loads(pickle.dumps(two)) is two
        True
    """
    if offset == 0:
        return UTC
    info = _tzinfos.get(offset)
    if info is None:
        info = _tzinfos.setdefault(offset, _FixedOffset(offset))
    return info