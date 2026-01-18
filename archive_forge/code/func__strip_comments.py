import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _strip_comments(line: bytes) -> bytes:
    comment_bytes = {ord(b'#'), ord(b';')}
    quote = ord(b'"')
    string_open = False
    for i, character in enumerate(bytearray(line)):
        if character == quote:
            string_open = not string_open
        elif not string_open and character in comment_bytes:
            return line[:i]
    return line