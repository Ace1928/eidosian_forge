import time as _time
import math as _math
import sys
from operator import index as _index
def _format_offset(off):
    s = ''
    if off is not None:
        if off.days < 0:
            sign = '-'
            off = -off
        else:
            sign = '+'
        hh, mm = divmod(off, timedelta(hours=1))
        mm, ss = divmod(mm, timedelta(minutes=1))
        s += '%s%02d:%02d' % (sign, hh, mm)
        if ss or ss.microseconds:
            s += ':%02d' % ss.seconds
            if ss.microseconds:
                s += '.%06d' % ss.microseconds
    return s