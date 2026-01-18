from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def _cretate_frame_tree(self, used_bytes=0, acquired_bytes=0):
    self._root.used_bytes += used_bytes
    self._root.acquired_bytes += acquired_bytes
    parent = self._root
    for depth, stackframe in enumerate(self._extract_stackframes()):
        if 0 < self._max_depth <= depth + 1:
            break
        memory_frame = self._add_frame(parent, stackframe)
        memory_frame.used_bytes += used_bytes
        memory_frame.acquired_bytes += acquired_bytes
        parent = memory_frame