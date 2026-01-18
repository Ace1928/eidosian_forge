import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _force_correct_text_reader(text_reader: t.IO[t.Any], encoding: t.Optional[str], errors: t.Optional[str], force_readable: bool=False) -> t.TextIO:
    return _force_correct_text_stream(text_reader, encoding, errors, _is_binary_reader, _find_binary_reader, force_readable=force_readable)