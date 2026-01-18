import logging
from contextlib import contextmanager
from vine.utils import wraps
from celery import states
from celery.backends.base import BaseBackend
from celery.exceptions import ImproperlyConfigured
from celery.utils.time import maybe_timedelta
from .models import Task, TaskExtended, TaskSet
from .session import SessionManager
def _update_result(self, task, result, state, traceback=None, request=None):
    meta = self._get_result_meta(result=result, state=state, traceback=traceback, request=request, format_date=False, encode=True)
    columns = [column.name for column in self.task_cls.__table__.columns if column.name not in {'id', 'task_id'}]
    for column in columns:
        value = meta.get(column)
        setattr(task, column, value)