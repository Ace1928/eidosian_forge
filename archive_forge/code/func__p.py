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
def _p(*args):
    """Factory function for unpickling pytz tzinfo instances.

    Just a wrapper around tzinfo.unpickler to save a few bytes in each pickle
    by shortening the path.
    """
    return unpickler(*args)