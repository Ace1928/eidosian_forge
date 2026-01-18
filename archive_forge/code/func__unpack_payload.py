import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
def _unpack_payload(b_data: memoryview, field_offset: int, unpack_func: typing.Optional[typing.Callable[[bytes], typing.Any]]=None) -> typing.Any:
    field_len = struct.unpack('<H', b_data[field_offset:field_offset + 2].tobytes())[0]
    if field_len:
        field_offset = struct.unpack('<I', b_data[field_offset + 4:field_offset + 8].tobytes())[0]
        b_value = b_data[field_offset:field_offset + field_len].tobytes()
        return unpack_func(b_value) if unpack_func else b_value