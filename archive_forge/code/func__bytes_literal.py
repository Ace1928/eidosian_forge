import re
from . import cursors, _mysql
from ._exceptions import (
def _bytes_literal(self, bs):
    assert isinstance(bs, (bytes, bytearray))
    x = self.string_literal(bs)
    if self._binary_prefix:
        return b'_binary' + x
    return x