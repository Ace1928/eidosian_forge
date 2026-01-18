import functools
import json
import multiprocessing
import os
import threading
from contextlib import contextmanager
from threading import Thread
from ._colorizer import Colorizer
from ._locks_machinery import create_handler_lock
@staticmethod
def _serialize_record(text, record):
    exception = record['exception']
    if exception is not None:
        exception = {'type': None if exception.type is None else exception.type.__name__, 'value': exception.value, 'traceback': bool(exception.traceback)}
    serializable = {'text': text, 'record': {'elapsed': {'repr': record['elapsed'], 'seconds': record['elapsed'].total_seconds()}, 'exception': exception, 'extra': record['extra'], 'file': {'name': record['file'].name, 'path': record['file'].path}, 'function': record['function'], 'level': {'icon': record['level'].icon, 'name': record['level'].name, 'no': record['level'].no}, 'line': record['line'], 'message': record['message'], 'module': record['module'], 'name': record['name'], 'process': {'id': record['process'].id, 'name': record['process'].name}, 'thread': {'id': record['thread'].id, 'name': record['thread'].name}, 'time': {'repr': record['time'], 'timestamp': record['time'].timestamp()}}}
    return json.dumps(serializable, default=str, ensure_ascii=False) + '\n'