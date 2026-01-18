import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _parse_region(self, offset, data):
    """Parse node data returned from a readv operation.

        :param offset: The byte offset the data starts at.
        :param data: The data to parse.
        """
    end = offset + len(data)
    high_parsed = offset
    while True:
        index = self._parsed_byte_index(high_parsed)
        if end < self._parsed_byte_map[index][1]:
            return
        high_parsed, last_segment = self._parse_segment(offset, data, end, index)
        if last_segment:
            return