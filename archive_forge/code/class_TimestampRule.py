from datetime import datetime
from datetime import timedelta
from datetime import timezone
from cloudsdk.google.protobuf import duration_pb2
from cloudsdk.google.protobuf import timestamp_pb2
from proto import datetime_helpers, utils
class TimestampRule:
    """A marshal between Python datetimes and protobuf timestamps.

    Note: Python datetimes are less precise than protobuf datetimes
    (microsecond vs. nanosecond level precision). If nanosecond-level
    precision matters, it is recommended to interact with the internal
    proto directly.
    """

    def to_python(self, value, *, absent: bool=None) -> datetime_helpers.DatetimeWithNanoseconds:
        if isinstance(value, timestamp_pb2.Timestamp):
            if absent:
                return None
            return datetime_helpers.DatetimeWithNanoseconds.from_timestamp_pb(value)
        return value

    def to_proto(self, value) -> timestamp_pb2.Timestamp:
        if isinstance(value, datetime_helpers.DatetimeWithNanoseconds):
            return value.timestamp_pb()
        if isinstance(value, datetime):
            return timestamp_pb2.Timestamp(seconds=int(value.timestamp()), nanos=value.microsecond * 1000)
        if isinstance(value, str):
            timestamp_value = timestamp_pb2.Timestamp()
            timestamp_value.FromJsonString(value=value)
            return timestamp_value
        return value