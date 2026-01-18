import time as _time
import math as _math
import sys
from operator import index as _index
def _check_tzinfo_arg(tz):
    if tz is not None and (not isinstance(tz, tzinfo)):
        raise TypeError('tzinfo argument must be None or of a tzinfo subclass')