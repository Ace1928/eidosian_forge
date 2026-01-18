import datetime
import time
class _FallbackLocalTimezone(datetime.tzinfo):

    def utcoffset(self, dt: datetime.datetime) -> datetime.timedelta:
        if self._isdst(dt):
            return DSTOFFSET
        else:
            return STDOFFSET

    def dst(self, dt: datetime.datetime) -> datetime.timedelta:
        if self._isdst(dt):
            return DSTDIFF
        else:
            return ZERO

    def tzname(self, dt: datetime.datetime) -> str:
        return time.tzname[self._isdst(dt)]

    def _isdst(self, dt: datetime.datetime) -> bool:
        tt = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.weekday(), 0, -1)
        stamp = time.mktime(tt)
        tt = time.localtime(stamp)
        return tt.tm_isdst > 0