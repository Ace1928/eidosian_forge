import sys
from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu import serialization
from kombu.exceptions import OperationalError
from kombu.utils.uuid import uuid
from celery import current_app, states
from celery._state import _task_stack
from celery.canvas import _chain, group, signature
from celery.exceptions import Ignore, ImproperlyConfigured, MaxRetriesExceededError, Reject, Retry
from celery.local import class_property
from celery.result import EagerResult, denied_join_result
from celery.utils import abstract
from celery.utils.functional import mattrgetter, maybe_list
from celery.utils.imports import instantiate
from celery.utils.nodenames import gethostname
from celery.utils.serialization import raise_with_context
from .annotations import resolve_all as resolve_all_annotations
from .registry import _unpickle_task_v2
from .utils import appstr
def _get_custom_headers(self, *args, **kwargs):
    headers = {}
    headers.update(*args, **kwargs)
    celery_keys = {*Context.__dict__.keys(), 'lang', 'task', 'argsrepr', 'kwargsrepr', 'compression'}
    for key in celery_keys:
        headers.pop(key, None)
    if not headers:
        return None
    return headers