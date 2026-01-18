from . import events
from . import exceptions
from . import tasks
class TaskGroup:
    """Asynchronous context manager for managing groups of tasks.

    Example use:

        async with asyncio.TaskGroup() as group:
            task1 = group.create_task(some_coroutine(...))
            task2 = group.create_task(other_coroutine(...))
        print("Both tasks have completed now.")

    All tasks are awaited when the context manager exits.

    Any exceptions other than `asyncio.CancelledError` raised within
    a task will cancel all remaining tasks and wait for them to exit.
    The exceptions are then combined and raised as an `ExceptionGroup`.
    """

    def __init__(self):
        self._entered = False
        self._exiting = False
        self._aborting = False
        self._loop = None
        self._parent_task = None
        self._parent_cancel_requested = False
        self._tasks = set()
        self._errors = []
        self._base_error = None
        self._on_completed_fut = None

    def __repr__(self):
        info = ['']
        if self._tasks:
            info.append(f'tasks={len(self._tasks)}')
        if self._errors:
            info.append(f'errors={len(self._errors)}')
        if self._aborting:
            info.append('cancelling')
        elif self._entered:
            info.append('entered')
        info_str = ' '.join(info)
        return f'<TaskGroup{info_str}>'

    async def __aenter__(self):
        if self._entered:
            raise RuntimeError(f'TaskGroup {self!r} has already been entered')
        if self._loop is None:
            self._loop = events.get_running_loop()
        self._parent_task = tasks.current_task(self._loop)
        if self._parent_task is None:
            raise RuntimeError(f'TaskGroup {self!r} cannot determine the parent task')
        self._entered = True
        return self

    async def __aexit__(self, et, exc, tb):
        self._exiting = True
        if exc is not None and self._is_base_error(exc) and (self._base_error is None):
            self._base_error = exc
        propagate_cancellation_error = exc if et is exceptions.CancelledError else None
        if self._parent_cancel_requested:
            if self._parent_task.uncancel() == 0:
                propagate_cancellation_error = None
        if et is not None:
            if not self._aborting:
                self._abort()
        while self._tasks:
            if self._on_completed_fut is None:
                self._on_completed_fut = self._loop.create_future()
            try:
                await self._on_completed_fut
            except exceptions.CancelledError as ex:
                if not self._aborting:
                    propagate_cancellation_error = ex
                    self._abort()
            self._on_completed_fut = None
        assert not self._tasks
        if self._base_error is not None:
            raise self._base_error
        if propagate_cancellation_error and (not self._errors):
            raise propagate_cancellation_error
        if et is not None and et is not exceptions.CancelledError:
            self._errors.append(exc)
        if self._errors:
            try:
                me = BaseExceptionGroup('unhandled errors in a TaskGroup', self._errors)
                raise me from None
            finally:
                self._errors = None

    def create_task(self, coro, *, name=None, context=None):
        """Create a new task in this group and return it.

        Similar to `asyncio.create_task`.
        """
        if not self._entered:
            raise RuntimeError(f'TaskGroup {self!r} has not been entered')
        if self._exiting and (not self._tasks):
            raise RuntimeError(f'TaskGroup {self!r} is finished')
        if self._aborting:
            raise RuntimeError(f'TaskGroup {self!r} is shutting down')
        if context is None:
            task = self._loop.create_task(coro)
        else:
            task = self._loop.create_task(coro, context=context)
        tasks._set_task_name(task, name)
        task.add_done_callback(self._on_task_done)
        self._tasks.add(task)
        return task

    def _is_base_error(self, exc: BaseException) -> bool:
        assert isinstance(exc, BaseException)
        return isinstance(exc, (SystemExit, KeyboardInterrupt))

    def _abort(self):
        self._aborting = True
        for t in self._tasks:
            if not t.done():
                t.cancel()

    def _on_task_done(self, task):
        self._tasks.discard(task)
        if self._on_completed_fut is not None and (not self._tasks):
            if not self._on_completed_fut.done():
                self._on_completed_fut.set_result(True)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        self._errors.append(exc)
        if self._is_base_error(exc) and self._base_error is None:
            self._base_error = exc
        if self._parent_task.done():
            self._loop.call_exception_handler({'message': f'Task {task!r} has errored out but its parent task {self._parent_task} is already completed', 'exception': exc, 'task': task})
            return
        if not self._aborting and (not self._parent_cancel_requested):
            self._abort()
            self._parent_cancel_requested = True
            self._parent_task.cancel()