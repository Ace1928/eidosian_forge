from . import events
from . import exceptions
from . import tasks
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