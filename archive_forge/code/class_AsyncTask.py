from enum import Enum
from queue import Queue
from threading import Thread
from typing import Callable, Optional, List
from .errors import AsyncTaskException
class AsyncTask:

    def __init__(self, func: Callable, *args, **kwargs):
        self.func: Callable = func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.result = None
        self.thread: Thread = None
        self.exception: Exception = None
        self.on_success = None
        self.on_error = None
        self.execute_on_caller = False

    def func_handler(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.status = TaskStatus.SUCCESS
            self.result = result
            if self.on_success is not None and (not self.execute_on_caller):
                self.on_success(self.result)
        except Exception as e:
            self.exception = e
            self.status = TaskStatus.FAILURE
            if self.on_error is not None and (not self.execute_on_caller):
                self.on_error(self.exception)
        global_task_manager.remove(self.thread)

    def run(self):
        child_thread = Thread(target=self.func_handler)
        self.thread = child_thread
        global_task_manager.put(child_thread)

    def wait(self):
        while not self.thread.is_alive():
            pass
        self.thread.join()
        if self.exception is not None:
            raise AsyncTaskException(str(self.exception), self.func.__name__)
        return self.result

    def subscribe(self, on_success: Callable, on_error: Callable, blocking: bool=False):
        if on_success is None or on_error is None:
            raise ValueError('Illegal argument. Callbacks on_success and on_error must not be null')
        self.on_success = on_success
        self.on_error = on_error
        self.execute_on_caller = blocking
        if blocking:
            while not self.thread.is_alive():
                pass
            self.thread.join()
            if self.exception is None:
                on_success(self.result)
            else:
                on_error(self.exception)