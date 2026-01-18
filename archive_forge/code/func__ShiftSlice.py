import array
import contextlib
import enum
import struct
def _ShiftSlice(s, offset, length):
    start = offset + (0 if s.start is None else s.start)
    stop = offset + (length if s.stop is None else s.stop)
    return slice(start, stop, s.step)