import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _parsed_byte_index(self, offset):
    """Return the index of the entry immediately before offset.

        e.g. if the parsed map has regions 0,10 and 11,12 parsed, meaning that
        there is one unparsed byte (the 11th, addressed as[10]). then:
        asking for 0 will return 0
        asking for 10 will return 0
        asking for 11 will return 1
        asking for 12 will return 1
        """
    key = (offset, 0)
    return self._find_index(self._parsed_byte_map, key)