import sys, os, unicodedata
import py
from py.builtin import text, bytes
def _update_chars_on_current_line(self, text_or_bytes):
    newline = b'\n' if isinstance(text_or_bytes, bytes) else '\n'
    current_line = text_or_bytes.rsplit(newline, 1)[-1]
    if isinstance(current_line, bytes):
        current_line = current_line.decode('utf-8', errors='replace')
    if newline in text_or_bytes:
        self._chars_on_current_line = len(current_line)
        self._width_of_current_line = get_line_width(current_line)
    else:
        self._chars_on_current_line += len(current_line)
        self._width_of_current_line += get_line_width(current_line)