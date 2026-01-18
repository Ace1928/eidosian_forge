import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def _fill_buffer(self, number):
    if not self._expired:
        while len(self._buffer) < number:
            try:
                self._buffer.append(next(self._stream))
            except StopIteration:
                self._expired = True
                break
    return bool(self._buffer)