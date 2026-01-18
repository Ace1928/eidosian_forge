from collections import deque
from numbers import Integral
from .messages.specs import SPEC_BY_STATUS, SYSEX_END, SYSEX_START
def _feed_status_byte(self, status):
    if status == SYSEX_END:
        if self._status == SYSEX_START:
            self._bytes.append(SYSEX_END)
            self._messages.append(self._bytes)
        self._status = 0
    elif 248 <= status <= 255:
        if self._status != SYSEX_START:
            self._status = 0
        if status in SPEC_BY_STATUS:
            self._messages.append([status])
    elif status in SPEC_BY_STATUS:
        spec = SPEC_BY_STATUS[status]
        if spec['length'] == 1:
            self._messages.append([status])
            self._status = 0
        else:
            self._status = status
            self._bytes = [status]
            self._len = spec['length']
    else:
        pass