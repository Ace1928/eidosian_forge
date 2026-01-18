import time
from contextlib import contextmanager
from functools import partial
from ssl import CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED
from urllib.parse import unquote
from kombu.utils.functional import retry_over_time
from kombu.utils.objects import cached_property
from kombu.utils.url import _parse_url, maybe_sanitize_url
from celery import states
from celery._state import task_join_will_block
from celery.canvas import maybe_signature
from celery.exceptions import BackendStoreError, ChordError, ImproperlyConfigured
from celery.result import GroupResult, allow_join_result
from celery.utils.functional import _regen, dictfilter
from celery.utils.log import get_logger
from celery.utils.time import humanize_seconds
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
from .base import BaseKeyValueStoreBackend
def _unpack_chord_result(self, tup, decode, EXCEPTION_STATES=states.EXCEPTION_STATES, PROPAGATE_STATES=states.PROPAGATE_STATES):
    _, tid, state, retval = decode(tup)
    if state in EXCEPTION_STATES:
        retval = self.exception_to_python(retval)
    if state in PROPAGATE_STATES:
        raise ChordError(f'Dependency {tid} raised {retval!r}')
    return retval