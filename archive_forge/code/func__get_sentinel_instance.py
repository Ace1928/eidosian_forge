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
def _get_sentinel_instance(self, **params):
    connparams = params.copy()
    hosts = connparams.pop('hosts')
    min_other_sentinels = self._transport_options.get('min_other_sentinels', 0)
    sentinel_kwargs = self._transport_options.get('sentinel_kwargs', {})
    sentinel_instance = self.sentinel.Sentinel([(cp['host'], cp['port']) for cp in hosts], min_other_sentinels=min_other_sentinels, sentinel_kwargs=sentinel_kwargs, **connparams)
    return sentinel_instance