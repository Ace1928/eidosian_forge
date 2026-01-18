from os import path
import sys
import traceback
from cupy.cuda import memory_hook
class MemoryFrame(object):
    """A single stack frame along with sum of memory usage at the frame.

    Attributes:
        stackframe (FrameSummary): stackframe from traceback.extract_stack().
        parent (MemoryFrame): parent frame, that is, caller.
        children (list of MemoryFrame): child frames, that is, callees.
        used_bytes (int): memory bytes that users used from CuPy memory pool.
        acquired_bytes (int): memory bytes that CuPy memory pool acquired
            from GPU device.
    """

    def __init__(self, parent, stackframe):
        self.stackframe = stackframe
        self.children = []
        self._set_parent(parent)
        self.used_bytes = 0
        self.acquired_bytes = 0

    def humanized_bytes(self):
        used_bytes = self._humanized_size(self.used_bytes)
        acquired_bytes = self._humanized_size(self.acquired_bytes)
        return (used_bytes, acquired_bytes)

    def _set_parent(self, parent):
        if parent and parent not in parent.children:
            self.parent = parent
            parent.children.append(self)

    def _humanized_size(self, size):
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E']:
            if size < 1024.0:
                return '%3.2f%sB' % (size, unit)
            size /= 1024.0
        return '%.2f%sB' % (size, 'Z')