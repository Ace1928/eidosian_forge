import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _escape_value(value: bytes) -> bytes:
    """Escape a value."""
    value = value.replace(b'\\', b'\\\\')
    value = value.replace(b'\r', b'\\r')
    value = value.replace(b'\n', b'\\n')
    value = value.replace(b'\t', b'\\t')
    value = value.replace(b'"', b'\\"')
    return value