import asyncio
import logging
import weakref
from ._asyncio_loop import get_running_loop, get_task_loop
class StandardSink:

    def __init__(self, handler):
        self._handler = handler

    def write(self, message):
        record = message.record
        message = str(message)
        exc = record['exception']
        record = logging.getLogger().makeRecord(record['name'], record['level'].no, record['file'].path, record['line'], message, (), (exc.type, exc.value, exc.traceback) if exc else None, record['function'], {'extra': record['extra']})
        if exc:
            record.exc_text = '\n'
        self._handler.handle(record)

    def stop(self):
        self._handler.close()

    def tasks_to_complete(self):
        return []