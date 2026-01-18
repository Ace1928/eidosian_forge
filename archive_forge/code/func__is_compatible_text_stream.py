import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _is_compatible_text_stream(stream: t.TextIO, encoding: t.Optional[str], errors: t.Optional[str]) -> bool:
    """Check if a stream's encoding and errors attributes are
    compatible with the desired values.
    """
    return _is_compat_stream_attr(stream, 'encoding', encoding) and _is_compat_stream_attr(stream, 'errors', errors)