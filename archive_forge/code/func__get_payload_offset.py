import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
def _get_payload_offset(b_data: memoryview, field_offsets: typing.List[int]) -> int:
    payload_offset = None
    for field_offset in field_offsets:
        offset = struct.unpack('<I', b_data[field_offset + 4:field_offset + 8].tobytes())[0]
        if not payload_offset or (offset and offset < payload_offset):
            payload_offset = offset
    return payload_offset or len(b_data)