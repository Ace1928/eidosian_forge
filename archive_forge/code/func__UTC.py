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
def _UTC():
    """Factory function for utc unpickling.

    Makes sure that unpickling a utc instance always returns the same
    module global.

    These examples belong in the UTC class above, but it is obscured; or in
    the README.rst, but we are not depending on Python 2.4 so integrating
    the README.rst examples with the unit tests is not trivial.

    >>> import datetime, pickle
    >>> dt = datetime.datetime(2005, 3, 1, 14, 13, 21, tzinfo=utc)
    >>> naive = dt.replace(tzinfo=None)
    >>> p = pickle.dumps(dt, 1)
    >>> naive_p = pickle.dumps(naive, 1)
    >>> len(p) - len(naive_p)
    17
    >>> new = pickle.loads(p)
    >>> new == dt
    True
    >>> new is dt
    False
    >>> new.tzinfo is dt.tzinfo
    True
    >>> utc is UTC is timezone('UTC')
    True
    >>> utc is timezone('GMT')
    False
    """
    return utc