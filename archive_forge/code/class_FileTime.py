import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
class FileTime(datetime.datetime):
    """Windows FILETIME structure.

    FILETIME structure representing number of 100-nanosecond intervals that have elapsed since January 1, 1601 UTC.
    This subclasses the datetime object to provide a similar interface but with the `nanosecond` attribute.

    Attrs:
        nanosecond (int): The number of nanoseconds (< 1000) in the FileTime. Note this only has a precision of up to
            100 nanoseconds.

    .. _FILETIME:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-dtyp/2c57429b-fdd4-488f-b5fc-9e4cf020fcdf
    """
    _EPOCH_FILETIME = 116444736000000000

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> 'FileTime':
        ns = 0
        if 'nanosecond' in kwargs:
            ns = kwargs.pop('nanosecond')
        dt = super(FileTime, cls).__new__(cls, *args, **kwargs)
        dt.nanosecond = ns
        return dt

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__()
        self.nanosecond = getattr(self, 'nanosecond', None) or 0

    @classmethod
    def now(cls, tz: typing.Optional[datetime.tzinfo]=None) -> 'FileTime':
        """Construct a FileTime from the current time and optional time zone info."""
        return FileTime.from_datetime(datetime.datetime.now(tz=tz))

    @classmethod
    def from_datetime(cls, dt: datetime.datetime, ns: int=0) -> 'FileTime':
        """Creates a FileTime object from a datetime object."""
        return FileTime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute, second=dt.second, microsecond=dt.microsecond, tzinfo=dt.tzinfo, nanosecond=ns)

    def __str__(self) -> str:
        """Displays the datetime in ISO 8601 including the 100th nanosecond internal like .NET does."""
        fraction_seconds = ''
        if self.microsecond or self.nanosecond:
            fraction_seconds = self.strftime('.%f')
            if self.nanosecond:
                fraction_seconds += str(self.nanosecond // 100)
        timezone = 'Z'
        if self.tzinfo:
            utc_offset = self.strftime('%z')
            timezone = '%s:%s' % (utc_offset[:3], utc_offset[3:])
        return '{0}-{1:02d}-{2:02d}T{3:02d}:{4:02d}:{5:02d}{6}{7}'.format(self.year, self.month, self.day, self.hour, self.minute, self.second, fraction_seconds, timezone)

    def pack(self) -> bytes:
        """Packs the structure to bytes."""
        utc_tz = datetime.timezone.utc
        utc_dt = typing.cast(datetime.datetime, self.replace(tzinfo=self.tzinfo if self.tzinfo else utc_tz))
        td = utc_dt.astimezone(utc_tz) - datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=utc_tz)
        epoch_time_ms = td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6
        ns100 = FileTime._EPOCH_FILETIME + epoch_time_ms * 10 + self.nanosecond // 100
        return struct.pack('<Q', ns100)

    @staticmethod
    def unpack(b_data: bytes) -> 'FileTime':
        """Unpacks the structure from bytes."""
        filetime = struct.unpack('<Q', b_data)[0]
        epoch_time_ms = (filetime - FileTime._EPOCH_FILETIME) // 10
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=epoch_time_ms)
        ns = int(filetime % 10) * 100
        return FileTime.from_datetime(dt, ns=ns)