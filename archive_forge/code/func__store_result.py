import logging
from contextlib import contextmanager
from vine.utils import wraps
from celery import states
from celery.backends.base import BaseBackend
from celery.exceptions import ImproperlyConfigured
from celery.utils.time import maybe_timedelta
from .models import Task, TaskExtended, TaskSet
from .session import SessionManager
@retry
def _store_result(self, task_id, result, state, traceback=None, request=None, **kwargs):
    """Store return value and state of an executed task."""
    session = self.ResultSession()
    with session_cleanup(session):
        task = list(session.query(self.task_cls).filter(self.task_cls.task_id == task_id))
        task = task and task[0]
        if not task:
            task = self.task_cls(task_id)
            task.task_id = task_id
            session.add(task)
            session.flush()
        self._update_result(task, result, state, traceback=traceback, request=request)
        session.commit()