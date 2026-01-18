import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _is_parsed(self, offset):
    """Returns True if offset has been parsed."""
    index = self._parsed_byte_index(offset)
    if index == len(self._parsed_byte_map):
        return offset < self._parsed_byte_map[index - 1][1]
    start, end = self._parsed_byte_map[index]
    return offset >= start and offset < end