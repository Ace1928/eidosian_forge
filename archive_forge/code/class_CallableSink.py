import asyncio
import logging
import weakref
from ._asyncio_loop import get_running_loop, get_task_loop
class CallableSink:

    def __init__(self, function):
        self._function = function

    def write(self, message):
        self._function(message)

    def stop(self):
        pass

    def tasks_to_complete(self):
        return []