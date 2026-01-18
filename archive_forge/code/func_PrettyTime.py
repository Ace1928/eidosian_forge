from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def PrettyTime(remaining_time):
    """Creates a standard version for a given remaining time in seconds.

  Created over using strftime because strftime seems to be
    more suitable for a datetime object, rather than just a number of
    seconds remaining.
  Args:
    remaining_time: The number of seconds remaining as a float, or a
      string/None value indicating time was not correctly calculated.
  Returns:
    if remaining_time is a valid float, %H:%M:%D time remaining format with
    the nearest integer from remaining_time (%H might be higher than 23).
    Else, it returns the same message it received.
  """
    remaining_time = int(round(remaining_time))
    hours = remaining_time // 3600
    if hours >= 100:
        return '%d+ hrs' % min(hours, 999)
    remaining_time -= 3600 * hours
    minutes = remaining_time // 60
    remaining_time -= 60 * minutes
    seconds = remaining_time
    return str('%02d' % hours) + ':' + str('%02d' % minutes) + ':' + str('%02d' % seconds)