import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _is_binary_reader(stream: t.IO[t.Any], default: bool=False) -> bool:
    try:
        return isinstance(stream.read(0), bytes)
    except Exception:
        return default