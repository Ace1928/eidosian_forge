import re
import time
import platform
from collections import OrderedDict
import six
def _time_left(stime, timeout):
    """
    Return time remaining since ``stime`` before given ``timeout``.

    This function assists determining the value of ``timeout`` for
    class method :meth:`~.Terminal.kbhit` and similar functions.

    :arg float stime: starting time for measurement
    :arg float timeout: timeout period, may be set to None to
       indicate no timeout (where None is always returned).
    :rtype: float or int
    :returns: time remaining as float. If no time is remaining,
       then the integer ``0`` is returned.
    """
    return max(0, timeout - (time.time() - stime)) if timeout else timeout