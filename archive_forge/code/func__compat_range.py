from __future__ import unicode_literals
import itertools
import struct
def _compat_range(start, end, step=1):
    assert step > 0
    i = start
    while i < end:
        yield i
        i += step