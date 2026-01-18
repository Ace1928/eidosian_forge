import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def _remove_allocated_block(self, block):
    arena, start, stop = block
    blocks = self._allocated_blocks[arena]
    blocks.remove((start, stop))
    if not blocks:
        self._discard_arena(arena)