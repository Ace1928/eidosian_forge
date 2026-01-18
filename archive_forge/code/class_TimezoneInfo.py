import datetime
import math
import re
class TimezoneInfo(datetime.tzinfo):

    def __init__(self, h, m):
        self._name = 'UTC'
        if h != 0 and m != 0:
            self._name += '%+03d:%2d' % (h, m)
        self._delta = datetime.timedelta(hours=h, minutes=math.copysign(m, h))

    def utcoffset(self, dt):
        return self._delta

    def tzname(self, dt):
        return self._name

    def dst(self, dt):
        return datetime.timedelta(0)