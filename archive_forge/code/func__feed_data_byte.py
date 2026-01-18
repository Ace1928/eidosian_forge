from collections import deque
from numbers import Integral
from .messages.specs import SPEC_BY_STATUS, SYSEX_END, SYSEX_START
def _feed_data_byte(self, byte):
    if self._status:
        self._bytes.append(byte)
        if len(self._bytes) == self._len:
            self._messages.append(self._bytes)
            self._status = 0
    else:
        pass