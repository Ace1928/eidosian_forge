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
def _save_group(self, group_id, result):
    """Store the result of an executed group."""
    session = self.ResultSession()
    with session_cleanup(session):
        group = self.taskset_cls(group_id, result)
        session.add(group)
        session.flush()
        session.commit()
        return result