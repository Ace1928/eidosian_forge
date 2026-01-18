from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
def _Percolate(f):
    int_value = int(f)
    fraction = round(round(f, 4) - int_value, 4)
    return (int_value, fraction)