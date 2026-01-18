import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def _malloc(self, size):
    i = bisect.bisect_left(self._lengths, size)
    if i == len(self._lengths):
        return self._new_arena(size)
    else:
        length = self._lengths[i]
        seq = self._len_to_seq[length]
        block = seq.pop()
        if not seq:
            del self._len_to_seq[length], self._lengths[i]
    arena, start, stop = block
    del self._start_to_block[arena, start]
    del self._stop_to_block[arena, stop]
    return block