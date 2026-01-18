from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict
def _utf8_byte_offset_to_character_offset(s: str, offset: int):
    byte_offset = 0
    char_offset = 0
    for char_offset, character in enumerate(s):
        byte_offset += 1
        codepoint = ord(character)
        if codepoint >= _utf8_with_4_bytes:
            byte_offset += 3
        elif codepoint >= _utf8_with_3_bytes:
            byte_offset += 2
        elif codepoint >= _utf8_with_2_bytes:
            byte_offset += 1
        if byte_offset > offset:
            break
    else:
        char_offset += 1
    return char_offset