import time
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
from _pydevd_bundle.pydevd_constants import get_thread_id
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import ObjectWrapper, wrap_attr
import pydevd_file_utils
from _pydev_bundle import pydev_log
import sys
from urllib.parse import quote
class AsyncioLogger:

    def __init__(self):
        self.task_mgr = NameManager('Task')
        self.coro_mgr = NameManager('Coro')
        self.start_time = cur_time()

    def get_task_id(self, frame):
        asyncio = sys.modules.get('asyncio')
        if asyncio is None:
            return None
        while frame is not None:
            if 'self' in frame.f_locals:
                self_obj = frame.f_locals['self']
                if isinstance(self_obj, asyncio.Task):
                    method_name = frame.f_code.co_name
                    if method_name == '_step':
                        return id(self_obj)
            frame = frame.f_back
        return None

    def log_event(self, frame):
        event_time = cur_time() - self.start_time
        if not hasattr(frame, 'f_back') or frame.f_back is None:
            return
        asyncio = sys.modules.get('asyncio')
        if asyncio is None:
            return
        back = frame.f_back
        if 'self' in frame.f_locals:
            self_obj = frame.f_locals['self']
            if isinstance(self_obj, asyncio.Task):
                method_name = frame.f_code.co_name
                if method_name == 'set_result':
                    task_id = id(self_obj)
                    task_name = self.task_mgr.get(str(task_id))
                    send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'thread', 'stop', frame.f_code.co_filename, frame.f_lineno, frame)
                method_name = back.f_code.co_name
                if method_name == '__init__':
                    task_id = id(self_obj)
                    task_name = self.task_mgr.get(str(task_id))
                    send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'thread', 'start', frame.f_code.co_filename, frame.f_lineno, frame)
            method_name = frame.f_code.co_name
            if isinstance(self_obj, asyncio.Lock):
                if method_name in ('acquire', 'release'):
                    task_id = self.get_task_id(frame)
                    task_name = self.task_mgr.get(str(task_id))
                    if method_name == 'acquire':
                        if not self_obj._waiters and (not self_obj.locked()):
                            send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'lock', method_name + '_begin', frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                        if self_obj.locked():
                            method_name += '_begin'
                        else:
                            method_name += '_end'
                    elif method_name == 'release':
                        method_name += '_end'
                    send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'lock', method_name, frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
            if isinstance(self_obj, asyncio.Queue):
                if method_name in ('put', 'get', '_put', '_get'):
                    task_id = self.get_task_id(frame)
                    task_name = self.task_mgr.get(str(task_id))
                    if method_name == 'put':
                        send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'lock', 'acquire_begin', frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                    elif method_name == '_put':
                        send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'lock', 'acquire_end', frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                        send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'lock', 'release', frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                    elif method_name == 'get':
                        back = frame.f_back
                        if back.f_code.co_name != 'send':
                            send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'lock', 'acquire_begin', frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                        else:
                            send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'lock', 'acquire_end', frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                            send_concurrency_message('asyncio_event', event_time, task_name, task_name, 'lock', 'release', frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))