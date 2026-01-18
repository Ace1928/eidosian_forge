import time as _time
import math as _math
import sys
from operator import index as _index
def _to_microseconds(self):
    return (self._days * (24 * 3600) + self._seconds) * 1000000 + self._microseconds