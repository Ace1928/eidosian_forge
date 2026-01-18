import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@property
def _payload_offset(self) -> int:
    """Gets the offset of the first payload value."""
    return _get_payload_offset(self._data, [12, 20, 28, 36, 44, 52])