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
def _reconnect_pubsub(self):
    self._pubsub = None
    self.backend.client.connection_pool.reset()
    if self.subscribed_to:
        metas = self.backend.client.mget(self.subscribed_to)
        metas = [meta for meta in metas if meta]
        for meta in metas:
            self.on_state_change(self._decode_result(meta), None)
    self._pubsub = self.backend.client.pubsub(ignore_subscribe_messages=True)
    if self.subscribed_to:
        self._pubsub.subscribe(*self.subscribed_to)
    else:
        self._pubsub.connection = self._pubsub.connection_pool.get_connection('pubsub', self._pubsub.shard_hint)
        self._pubsub.connection.register_connect_callback(self._pubsub.on_connect)