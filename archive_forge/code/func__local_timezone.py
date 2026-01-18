import time as _time
import math as _math
import sys
from operator import index as _index
def _local_timezone(self):
    if self.tzinfo is None:
        ts = self._mktime()
    else:
        ts = (self - _EPOCH) // timedelta(seconds=1)
    localtm = _time.localtime(ts)
    local = datetime(*localtm[:6])
    gmtoff = localtm.tm_gmtoff
    zone = localtm.tm_zone
    return timezone(timedelta(seconds=gmtoff), zone)