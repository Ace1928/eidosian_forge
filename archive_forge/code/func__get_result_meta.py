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
def _get_result_meta(self, result, state, traceback, request, format_date=True, encode=False):
    if state in self.READY_STATES:
        date_done = datetime.now(timezone.utc)
        if format_date:
            date_done = date_done.isoformat()
    else:
        date_done = None
    meta = {'status': state, 'result': result, 'traceback': traceback, 'children': self.current_task_children(request), 'date_done': date_done}
    if request and getattr(request, 'group', None):
        meta['group_id'] = request.group
    if request and getattr(request, 'parent_id', None):
        meta['parent_id'] = request.parent_id
    if self.app.conf.find_value_for_key('extended', 'result'):
        if request:
            request_meta = {'name': getattr(request, 'task', None), 'args': getattr(request, 'args', None), 'kwargs': getattr(request, 'kwargs', None), 'worker': getattr(request, 'hostname', None), 'retries': getattr(request, 'retries', None), 'queue': request.delivery_info.get('routing_key') if hasattr(request, 'delivery_info') and request.delivery_info else None}
            if getattr(request, 'stamps', None):
                request_meta['stamped_headers'] = request.stamped_headers
                request_meta.update(request.stamps)
            if encode:
                encode_needed_fields = {'args', 'kwargs'}
                for field in encode_needed_fields:
                    value = request_meta[field]
                    encoded_value = self.encode(value)
                    request_meta[field] = ensure_bytes(encoded_value)
            meta.update(request_meta)
    return meta