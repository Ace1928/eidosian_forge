from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import calendar
import datetime
import re
import time
from six.moves import map  # pylint: disable=redefined-builtin
def CurrentTimeSec():
    """Returns a float of the current time in seconds."""
    return time.time()