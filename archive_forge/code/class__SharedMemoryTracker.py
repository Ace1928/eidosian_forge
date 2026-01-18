import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
class _SharedMemoryTracker:
    """Manages one or more shared memory segments."""

    def __init__(self, name, segment_names=[]):
        self.shared_memory_context_name = name
        self.segment_names = segment_names

    def register_segment(self, segment_name):
        """Adds the supplied shared memory block name to tracker."""
        util.debug(f'Register segment {segment_name!r} in pid {getpid()}')
        self.segment_names.append(segment_name)

    def destroy_segment(self, segment_name):
        """Calls unlink() on the shared memory block with the supplied name
            and removes it from the list of blocks being tracked."""
        util.debug(f'Destroy segment {segment_name!r} in pid {getpid()}')
        self.segment_names.remove(segment_name)
        segment = shared_memory.SharedMemory(segment_name)
        segment.close()
        segment.unlink()

    def unlink(self):
        """Calls destroy_segment() on all tracked shared memory blocks."""
        for segment_name in self.segment_names[:]:
            self.destroy_segment(segment_name)

    def __del__(self):
        util.debug(f'Call {self.__class__.__name__}.__del__ in {getpid()}')
        self.unlink()

    def __getstate__(self):
        return (self.shared_memory_context_name, self.segment_names)

    def __setstate__(self, state):
        self.__init__(*state)