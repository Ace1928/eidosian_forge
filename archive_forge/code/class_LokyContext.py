import os
import sys
import math
import subprocess
import traceback
import warnings
import multiprocessing as mp
from multiprocessing import get_context as mp_get_context
from multiprocessing.context import BaseContext
from .process import LokyProcess, LokyInitMainProcess
class LokyContext(BaseContext):
    """Context relying on the LokyProcess."""
    _name = 'loky'
    Process = LokyProcess
    cpu_count = staticmethod(cpu_count)

    def Queue(self, maxsize=0, reducers=None):
        """Returns a queue object"""
        from .queues import Queue
        return Queue(maxsize, reducers=reducers, ctx=self.get_context())

    def SimpleQueue(self, reducers=None):
        """Returns a queue object"""
        from .queues import SimpleQueue
        return SimpleQueue(reducers=reducers, ctx=self.get_context())
    if sys.platform != 'win32':
        'For Unix platform, use our custom implementation of synchronize\n        ensuring that we use the loky.backend.resource_tracker to clean-up\n        the semaphores in case of a worker crash.\n        '

        def Semaphore(self, value=1):
            """Returns a semaphore object"""
            from .synchronize import Semaphore
            return Semaphore(value=value)

        def BoundedSemaphore(self, value):
            """Returns a bounded semaphore object"""
            from .synchronize import BoundedSemaphore
            return BoundedSemaphore(value)

        def Lock(self):
            """Returns a lock object"""
            from .synchronize import Lock
            return Lock()

        def RLock(self):
            """Returns a recurrent lock object"""
            from .synchronize import RLock
            return RLock()

        def Condition(self, lock=None):
            """Returns a condition object"""
            from .synchronize import Condition
            return Condition(lock)

        def Event(self):
            """Returns an event object"""
            from .synchronize import Event
            return Event()