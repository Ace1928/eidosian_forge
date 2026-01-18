import re
from typing import List, Optional, Tuple
class _PushbackSequence:

    def __init__(self, orig):
        self._iter = iter(orig)
        self._pushback_buffer = []

    def __next__(self):
        if len(self._pushback_buffer) > 0:
            return self._pushback_buffer.pop()
        else:
            return next(self._iter)
    next = __next__

    def pushback(self, char):
        self._pushback_buffer.append(char)

    def __iter__(self):
        return self