import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _find_binary_writer(stream: t.IO[t.Any]) -> t.Optional[t.BinaryIO]:
    if _is_binary_writer(stream, False):
        return t.cast(t.BinaryIO, stream)
    buf = getattr(stream, 'buffer', None)
    if buf is not None and _is_binary_writer(buf, True):
        return t.cast(t.BinaryIO, buf)
    return None