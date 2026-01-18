from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def MakeHumanReadable(num):
    """Generates human readable string for a number of bytes.

  Args:
    num: The number, in bytes.

  Returns:
    A string form of the number using size abbreviations (KiB, MiB, etc.).
  """
    i, rounded_val = _RoundToNearestExponent(num)
    return '%g %s' % (rounded_val, _EXP_STRINGS[i][1])