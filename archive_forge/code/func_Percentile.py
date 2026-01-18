from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def Percentile(values, percent, key=lambda x: x):
    """Find the percentile of a list of values.

  Taken from: http://code.activestate.com/recipes/511478/

  Args:
    values: a list of numeric values. Note that the values MUST BE already
            sorted.
    percent: a float value from 0.0 to 1.0.
    key: optional key function to compute value from each element of the list
         of values.

  Returns:
    The percentile of the values.
  """
    if not values:
        return None
    k = (len(values) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(values[int(k)])
    d0 = key(values[int(f)]) * (c - k)
    d1 = key(values[int(c)]) * (k - f)
    return d0 + d1