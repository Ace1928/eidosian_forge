from aniso8601.builders import DatetimeTuple, DateTuple, TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.compat import is_string
from aniso8601.date import parse_date
from aniso8601.duration import parse_duration
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import IntervalResolution
from aniso8601.time import parse_datetime, parse_time
def _get_interval_component_resolution(componenttuple):
    if type(componenttuple) is DateTuple:
        if componenttuple.DDD is not None:
            return IntervalResolution.Ordinal
        if componenttuple.D is not None:
            return IntervalResolution.Weekday
        if componenttuple.Www is not None:
            return IntervalResolution.Week
        if componenttuple.DD is not None:
            return IntervalResolution.Day
        if componenttuple.MM is not None:
            return IntervalResolution.Month
        return IntervalResolution.Year
    elif type(componenttuple) is DatetimeTuple:
        if componenttuple.time.ss is not None:
            return IntervalResolution.Seconds
        if componenttuple.time.mm is not None:
            return IntervalResolution.Minutes
        return IntervalResolution.Hours
    if componenttuple.TnS is not None:
        return IntervalResolution.Seconds
    if componenttuple.TnM is not None:
        return IntervalResolution.Minutes
    if componenttuple.TnH is not None:
        return IntervalResolution.Hours
    if componenttuple.PnD is not None:
        return IntervalResolution.Day
    if componenttuple.PnW is not None:
        return IntervalResolution.Week
    if componenttuple.PnM is not None:
        return IntervalResolution.Month
    return IntervalResolution.Year