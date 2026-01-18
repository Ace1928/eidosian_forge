import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _force_correct_text_writer(text_writer: t.IO[t.Any], encoding: t.Optional[str], errors: t.Optional[str], force_writable: bool=False) -> t.TextIO:
    return _force_correct_text_stream(text_writer, encoding, errors, _is_binary_writer, _find_binary_writer, force_writable=force_writable)