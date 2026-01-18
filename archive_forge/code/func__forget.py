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
def _forget(self, task_id):
    """Forget about result."""
    session = self.ResultSession()
    with session_cleanup(session):
        session.query(self.task_cls).filter(self.task_cls.task_id == task_id).delete()
        session.commit()