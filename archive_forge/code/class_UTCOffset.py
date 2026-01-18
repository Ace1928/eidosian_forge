import datetime
class UTCOffset(datetime.tzinfo):

    def __init__(self, name=None, minutes=None):
        self._name = name
        if minutes is not None:
            self._utcdelta = datetime.timedelta(minutes=minutes)
        else:
            self._utcdelta = None

    def __repr__(self):
        if self._utcdelta >= datetime.timedelta(hours=0):
            return '+{0} UTC'.format(self._utcdelta)
        correcteddays = abs(self._utcdelta.days + 1)
        deltaseconds = 24 * 60 * 60 - self._utcdelta.seconds
        days, remainder = divmod(deltaseconds, 24 * 60 * 60)
        hours, remainder = divmod(remainder, 1 * 60 * 60)
        minutes, seconds = divmod(remainder, 1 * 60)
        correcteddays += days
        if correcteddays == 0:
            return '-{0}:{1:02}:{2:02} UTC'.format(hours, minutes, seconds)
        elif correcteddays == 1:
            return '-1 day, {0}:{1:02}:{2:02} UTC'.format(hours, minutes, seconds)
        return '-{0} days, {1}:{2:02}:{3:02} UTC'.format(correcteddays, hours, minutes, seconds)

    def utcoffset(self, dt):
        return self._utcdelta

    def tzname(self, dt):
        return self._name

    def dst(self, dt):
        return datetime.timedelta(0)