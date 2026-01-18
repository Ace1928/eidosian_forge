from celery import Task
from celery.result import AsyncResult
class AbortableTask(Task):
    """Task that can be aborted.

    This serves as a base class for all :class:`Task`'s
    that support aborting during execution.

    All subclasses of :class:`AbortableTask` must call the
    :meth:`is_aborted` method periodically and act accordingly when
    the call evaluates to :const:`True`.
    """
    abstract = True

    def AsyncResult(self, task_id):
        """Return the accompanying AbortableAsyncResult instance."""
        return AbortableAsyncResult(task_id, backend=self.backend)

    def is_aborted(self, **kwargs):
        """Return true if task is aborted.

        Checks against the backend whether this
        :class:`AbortableAsyncResult` is :const:`ABORTED`.

        Always return :const:`False` in case the `task_id` parameter
        refers to a regular (non-abortable) :class:`Task`.

        Be aware that invoking this method will cause a hit in the
        backend (for example a database query), so find a good balance
        between calling it regularly (for responsiveness), but not too
        often (for performance).
        """
        task_id = kwargs.get('task_id', self.request.id)
        result = self.AsyncResult(task_id)
        if not isinstance(result, AbortableAsyncResult):
            return False
        return result.is_aborted()