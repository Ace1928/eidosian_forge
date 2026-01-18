import logging
from contextlib import contextmanager
from vine.utils import wraps
from celery import states
from celery.backends.base import BaseBackend
from celery.exceptions import ImproperlyConfigured
from celery.utils.time import maybe_timedelta
from .models import Task, TaskExtended, TaskSet
from .session import SessionManager
def ResultSession(self, session_manager=SessionManager()):
    return session_manager.session_factory(dburi=self.url, short_lived_sessions=self.short_lived_sessions, **self.engine_options)