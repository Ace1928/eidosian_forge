import time as _time
import math as _math
import sys
from operator import index as _index
def _check_tzname(name):
    if name is not None and (not isinstance(name, str)):
        raise TypeError("tzinfo.tzname() must return None or string, not '%s'" % type(name))