import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def _discard_arena(self, arena):
    length = arena.size
    if length < self._DISCARD_FREE_SPACE_LARGER_THAN:
        return
    blocks = self._allocated_blocks.pop(arena)
    assert not blocks
    del self._start_to_block[arena, 0]
    del self._stop_to_block[arena, length]
    self._arenas.remove(arena)
    seq = self._len_to_seq[length]
    seq.remove((arena, 0, length))
    if not seq:
        del self._len_to_seq[length]
        self._lengths.remove(length)