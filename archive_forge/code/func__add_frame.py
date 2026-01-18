from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def _add_frame(self, parent, stackframe):
    key = self._key_frame(parent, stackframe)
    if key in self._memory_frames:
        memory_frame = self._memory_frames[key]
    else:
        memory_frame = MemoryFrame(parent, stackframe)
        self._memory_frames[key] = memory_frame
    return memory_frame