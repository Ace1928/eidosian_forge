import sys
import time
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import partial
from weakref import WeakValueDictionary
from billiard.einfo import ExceptionInfo
from kombu.serialization import dumps, loads, prepare_accept_content
from kombu.serialization import registry as serializer_registry
from kombu.utils.encoding import bytes_to_str, ensure_bytes
from kombu.utils.url import maybe_sanitize_url
import celery.exceptions
from celery import current_app, group, maybe_signature, states
from celery._state import get_current_task
from celery.app.task import Context
from celery.exceptions import (BackendGetMetaError, BackendStoreError, ChordError, ImproperlyConfigured,
from celery.result import GroupResult, ResultBase, ResultSet, allow_join_result, result_from_tuple
from celery.utils.collections import BufferMap
from celery.utils.functional import LRUCache, arity_greater
from celery.utils.log import get_logger
from celery.utils.serialization import (create_exception_cls, ensure_serializable, get_pickleable_exception,
from celery.utils.time import get_exponential_backoff_interval
def _call_task_errbacks(self, request, exc, traceback):
    old_signature = []
    for errback in request.errbacks:
        errback = self.app.signature(errback)
        if not errback._app:
            errback._app = self.app
        try:
            if hasattr(errback.type, '__header__') and (not isinstance(errback.type.__header__, partial)) and arity_greater(errback.type.__header__, 1):
                errback(request, exc, traceback)
            else:
                old_signature.append(errback)
        except NotRegistered:
            old_signature.append(errback)
    if old_signature:
        task_id = request.id
        root_id = request.root_id or task_id
        g = group(old_signature, app=self.app)
        if self.app.conf.task_always_eager or request.delivery_info.get('is_eager', False):
            g.apply((task_id,), parent_id=task_id, root_id=root_id)
        else:
            g.apply_async((task_id,), parent_id=task_id, root_id=root_id)