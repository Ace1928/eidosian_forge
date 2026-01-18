import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _force_correct_text_stream(text_stream: t.IO[t.Any], encoding: t.Optional[str], errors: t.Optional[str], is_binary: t.Callable[[t.IO[t.Any], bool], bool], find_binary: t.Callable[[t.IO[t.Any]], t.Optional[t.BinaryIO]], force_readable: bool=False, force_writable: bool=False) -> t.TextIO:
    if is_binary(text_stream, False):
        binary_reader = t.cast(t.BinaryIO, text_stream)
    else:
        text_stream = t.cast(t.TextIO, text_stream)
        if _is_compatible_text_stream(text_stream, encoding, errors) and (not (encoding is None and _stream_is_misconfigured(text_stream))):
            return text_stream
        possible_binary_reader = find_binary(text_stream)
        if possible_binary_reader is None:
            return text_stream
        binary_reader = possible_binary_reader
    if errors is None:
        errors = 'replace'
    return _make_text_stream(binary_reader, encoding, errors, force_readable=force_readable, force_writable=force_writable)