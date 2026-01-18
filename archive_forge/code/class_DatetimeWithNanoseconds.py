import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
class DatetimeWithNanoseconds(datetime.datetime):
    """Track nanosecond in addition to normal datetime attrs.

    Nanosecond can be passed only as a keyword argument.
    """
    __slots__ = ('_nanosecond',)

    def __new__(cls, *args, **kw):
        nanos = kw.pop('nanosecond', 0)
        if nanos > 0:
            if 'microsecond' in kw:
                raise TypeError("Specify only one of 'microsecond' or 'nanosecond'")
            kw['microsecond'] = nanos // 1000
        inst = datetime.datetime.__new__(cls, *args, **kw)
        inst._nanosecond = nanos or 0
        return inst

    @property
    def nanosecond(self):
        """Read-only: nanosecond precision."""
        return self._nanosecond

    def rfc3339(self):
        """Return an RFC3339-compliant timestamp.

        Returns:
            (str): Timestamp string according to RFC3339 spec.
        """
        if self._nanosecond == 0:
            return to_rfc3339(self)
        nanos = str(self._nanosecond).rjust(9, '0').rstrip('0')
        return '{}.{}Z'.format(self.strftime(_RFC3339_NO_FRACTION), nanos)

    @classmethod
    def from_rfc3339(cls, stamp):
        """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (str): RFC3339 stamp, with up to nanosecond precision

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp string

        Raises:
            ValueError: if `stamp` does not match the expected format
        """
        with_nanos = _RFC3339_NANOS.match(stamp)
        if with_nanos is None:
            raise ValueError('Timestamp: {}, does not match pattern: {}'.format(stamp, _RFC3339_NANOS.pattern))
        bare = datetime.datetime.strptime(with_nanos.group('no_fraction'), _RFC3339_NO_FRACTION)
        fraction = with_nanos.group('nanos')
        if fraction is None:
            nanos = 0
        else:
            scale = 9 - len(fraction)
            nanos = int(fraction) * 10 ** scale
        return cls(bare.year, bare.month, bare.day, bare.hour, bare.minute, bare.second, nanosecond=nanos, tzinfo=datetime.timezone.utc)

    def timestamp_pb(self):
        """Return a timestamp message.

        Returns:
            (:class:`~google.protobuf.timestamp_pb2.Timestamp`): Timestamp message
        """
        inst = self if self.tzinfo is not None else self.replace(tzinfo=datetime.timezone.utc)
        delta = inst - _UTC_EPOCH
        seconds = int(delta.total_seconds())
        nanos = self._nanosecond or self.microsecond * 1000
        return timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)

    @classmethod
    def from_timestamp_pb(cls, stamp):
        """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (:class:`~google.protobuf.timestamp_pb2.Timestamp`): timestamp message

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp message
        """
        microseconds = int(stamp.seconds * 1000000.0)
        bare = from_microseconds(microseconds)
        return cls(bare.year, bare.month, bare.day, bare.hour, bare.minute, bare.second, nanosecond=stamp.nanos, tzinfo=datetime.timezone.utc)