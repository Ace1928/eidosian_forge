import asyncio
import asyncio.coroutines
import contextvars
import functools
import inspect
import os
import sys
import threading
import warnings
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
from .current_thread_executor import CurrentThreadExecutor
from .local import Local
class SyncToAsync(Generic[_P, _R]):
    """
    Utility class which turns a synchronous callable into an awaitable that
    runs in a threadpool. It also sets a threadlocal inside the thread so
    calls to AsyncToSync can escape it.

    If thread_sensitive is passed, the code will run in the same thread as any
    outer code. This is needed for underlying Python code that is not
    threadsafe (for example, code which handles SQLite database connections).

    If the outermost program is async (i.e. SyncToAsync is outermost), then
    this will be a dedicated single sub-thread that all sync code runs in,
    one after the other. If the outermost program is sync (i.e. AsyncToSync is
    outermost), this will just be the main thread. This is achieved by idling
    with a CurrentThreadExecutor while AsyncToSync is blocking its sync parent,
    rather than just blocking.

    If executor is passed in, that will be used instead of the loop's default executor.
    In order to pass in an executor, thread_sensitive must be set to False, otherwise
    a TypeError will be raised.
    """
    threadlocal = threading.local()
    single_thread_executor = ThreadPoolExecutor(max_workers=1)
    thread_sensitive_context: 'contextvars.ContextVar[ThreadSensitiveContext]' = contextvars.ContextVar('thread_sensitive_context')
    deadlock_context: 'contextvars.ContextVar[bool]' = contextvars.ContextVar('deadlock_context')
    context_to_thread_executor: 'weakref.WeakKeyDictionary[ThreadSensitiveContext, ThreadPoolExecutor]' = weakref.WeakKeyDictionary()

    def __init__(self, func: Callable[_P, _R], thread_sensitive: bool=True, executor: Optional['ThreadPoolExecutor']=None) -> None:
        if not callable(func) or iscoroutinefunction(func) or iscoroutinefunction(getattr(func, '__call__', func)):
            raise TypeError('sync_to_async can only be applied to sync functions.')
        self.func = func
        functools.update_wrapper(self, func)
        self._thread_sensitive = thread_sensitive
        markcoroutinefunction(self)
        if thread_sensitive and executor is not None:
            raise TypeError('executor must not be set when thread_sensitive is True')
        self._executor = executor
        try:
            self.__self__ = func.__self__
        except AttributeError:
            pass

    async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        __traceback_hide__ = True
        loop = asyncio.get_running_loop()
        if self._thread_sensitive:
            current_thread_executor = getattr(AsyncToSync.executors, 'current', None)
            if current_thread_executor:
                executor = current_thread_executor
            elif self.thread_sensitive_context.get(None):
                thread_sensitive_context = self.thread_sensitive_context.get()
                if thread_sensitive_context in self.context_to_thread_executor:
                    executor = self.context_to_thread_executor[thread_sensitive_context]
                else:
                    executor = ThreadPoolExecutor(max_workers=1)
                    self.context_to_thread_executor[thread_sensitive_context] = executor
            elif loop in AsyncToSync.loop_thread_executors:
                executor = AsyncToSync.loop_thread_executors[loop]
            elif self.deadlock_context.get(False):
                raise RuntimeError('Single thread executor already being used, would deadlock')
            else:
                executor = self.single_thread_executor
                self.deadlock_context.set(True)
        else:
            executor = self._executor
        context = contextvars.copy_context()
        child = functools.partial(self.func, *args, **kwargs)
        func = context.run
        task_context: List[asyncio.Task[Any]] = []
        exec_coro = loop.run_in_executor(executor, functools.partial(self.thread_handler, loop, sys.exc_info(), task_context, func, child))
        ret: _R
        try:
            ret = await asyncio.shield(exec_coro)
        except asyncio.CancelledError:
            cancel_parent = True
            try:
                task = task_context[0]
                task.cancel()
                try:
                    await task
                    cancel_parent = False
                except asyncio.CancelledError:
                    pass
            except IndexError:
                pass
            if exec_coro.done():
                raise
            if cancel_parent:
                exec_coro.cancel()
            ret = await exec_coro
        finally:
            _restore_context(context)
            self.deadlock_context.set(False)
        return ret

    def __get__(self, parent: Any, objtype: Any) -> Callable[_P, Coroutine[Any, Any, _R]]:
        """
        Include self for methods
        """
        func = functools.partial(self.__call__, parent)
        return functools.update_wrapper(func, self.func)

    def thread_handler(self, loop, exc_info, task_context, func, *args, **kwargs):
        """
        Wraps the sync application with exception handling.
        """
        __traceback_hide__ = True
        self.threadlocal.main_event_loop = loop
        self.threadlocal.main_event_loop_pid = os.getpid()
        self.threadlocal.task_context = task_context
        if exc_info[1]:
            try:
                raise exc_info[1]
            except BaseException:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)