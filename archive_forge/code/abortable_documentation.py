from celery import Task
from celery.result import AsyncResult
Return true if task is aborted.

        Checks against the backend whether this
        :class:`AbortableAsyncResult` is :const:`ABORTED`.

        Always return :const:`False` in case the `task_id` parameter
        refers to a regular (non-abortable) :class:`Task`.

        Be aware that invoking this method will cause a hit in the
        backend (for example a database query), so find a good balance
        between calling it regularly (for responsiveness), but not too
        often (for performance).
        