import time as _time
import math as _math
import sys
from operator import index as _index
def _getstate(self, protocol=3):
    yhi, ylo = divmod(self._year, 256)
    us2, us3 = divmod(self._microsecond, 256)
    us1, us2 = divmod(us2, 256)
    m = self._month
    if self._fold and protocol > 3:
        m += 128
    basestate = bytes([yhi, ylo, m, self._day, self._hour, self._minute, self._second, us1, us2, us3])
    if self._tzinfo is None:
        return (basestate,)
    else:
        return (basestate, self._tzinfo)