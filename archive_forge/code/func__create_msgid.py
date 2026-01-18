from collections import deque
import select
import msgpack
def _create_msgid(self):
    this_id = self._next_msgid
    self._next_msgid = (self._next_msgid + 1) % 4294967295
    return this_id