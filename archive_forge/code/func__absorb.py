import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def _absorb(self, block):
    arena, start, stop = block
    del self._start_to_block[arena, start]
    del self._stop_to_block[arena, stop]
    length = stop - start
    seq = self._len_to_seq[length]
    seq.remove(block)
    if not seq:
        del self._len_to_seq[length]
        self._lengths.remove(length)
    return (start, stop)