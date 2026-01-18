from __future__ import absolute_import
import sys
import time
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk._compat import reraise
from sentry_sdk._functools import wraps
from sentry_sdk.crons import capture_checkin, MonitorStatus
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import BAGGAGE_HEADER_NAME, TRANSACTION_SOURCE_TASK
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _wrap_tracer(task, f):

    @wraps(f)
    def _inner(*args, **kwargs):
        hub = Hub.current
        if hub.get_integration(CeleryIntegration) is None:
            return f(*args, **kwargs)
        with hub.push_scope() as scope:
            scope._name = 'celery'
            scope.clear_breadcrumbs()
            scope.add_event_processor(_make_event_processor(task, *args, **kwargs))
            transaction = None
            with capture_internal_exceptions():
                transaction = continue_trace(args[3].get('headers') or {}, op=OP.QUEUE_TASK_CELERY, name='unknown celery task', source=TRANSACTION_SOURCE_TASK)
                transaction.name = task.name
                transaction.set_status('ok')
            if transaction is None:
                return f(*args, **kwargs)
            with hub.start_transaction(transaction, custom_sampling_context={'celery_job': {'task': task.name, 'args': list(args[1]), 'kwargs': args[2]}}):
                return f(*args, **kwargs)
    return _inner