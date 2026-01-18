from aniso8601.builders import DatetimeTuple, DateTuple, TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.compat import is_string
from aniso8601.date import parse_date
from aniso8601.duration import parse_duration
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import IntervalResolution
from aniso8601.time import parse_datetime, parse_time
def _get_interval_resolution(intervaltuple):
    if intervaltuple.start is not None and intervaltuple.end is not None:
        return max(_get_interval_component_resolution(intervaltuple.start), _get_interval_component_resolution(intervaltuple.end))
    if intervaltuple.start is not None and intervaltuple.duration is not None:
        return max(_get_interval_component_resolution(intervaltuple.start), _get_interval_component_resolution(intervaltuple.duration))
    return max(_get_interval_component_resolution(intervaltuple.end), _get_interval_component_resolution(intervaltuple.duration))