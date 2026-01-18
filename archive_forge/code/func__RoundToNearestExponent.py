from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def _RoundToNearestExponent(num):
    i = 0
    while i + 1 < len(_EXP_STRINGS) and num >= 2 ** _EXP_STRINGS[i + 1][0]:
        i += 1
    return (i, round(float(num) / 2.0 ** _EXP_STRINGS[i][0], 2))