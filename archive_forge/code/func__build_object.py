import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def _build_object(cls, parsetuple):
    if type(parsetuple) is DateTuple:
        return cls.build_date(YYYY=parsetuple.YYYY, MM=parsetuple.MM, DD=parsetuple.DD, Www=parsetuple.Www, D=parsetuple.D, DDD=parsetuple.DDD)
    if type(parsetuple) is TimeTuple:
        return cls.build_time(hh=parsetuple.hh, mm=parsetuple.mm, ss=parsetuple.ss, tz=parsetuple.tz)
    if type(parsetuple) is DatetimeTuple:
        return cls.build_datetime(parsetuple.date, parsetuple.time)
    if type(parsetuple) is DurationTuple:
        return cls.build_duration(PnY=parsetuple.PnY, PnM=parsetuple.PnM, PnW=parsetuple.PnW, PnD=parsetuple.PnD, TnH=parsetuple.TnH, TnM=parsetuple.TnM, TnS=parsetuple.TnS)
    if type(parsetuple) is IntervalTuple:
        return cls.build_interval(start=parsetuple.start, end=parsetuple.end, duration=parsetuple.duration)
    if type(parsetuple) is RepeatingIntervalTuple:
        return cls.build_repeating_interval(R=parsetuple.R, Rnn=parsetuple.Rnn, interval=parsetuple.interval)
    return cls.build_timezone(negative=parsetuple.negative, Z=parsetuple.Z, hh=parsetuple.hh, mm=parsetuple.mm, name=parsetuple.name)