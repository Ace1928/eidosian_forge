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
class SentinelBackend(RedisBackend):
    """Redis sentinel task result store."""
    _SERVER_URI_SEPARATOR = ';'
    sentinel = getattr(redis, 'sentinel', None)
    connection_class_ssl = SentinelManagedSSLConnection if sentinel else None

    def __init__(self, *args, **kwargs):
        if self.sentinel is None:
            raise ImproperlyConfigured(E_REDIS_SENTINEL_MISSING.strip())
        super().__init__(*args, **kwargs)

    def as_uri(self, include_password=False):
        """Return the server addresses as URIs, sanitizing the password or not."""
        if include_password:
            return super().as_uri(include_password=include_password)
        uri_chunks = (maybe_sanitize_url(chunk) for chunk in (self.url or '').split(self._SERVER_URI_SEPARATOR))
        return self._SERVER_URI_SEPARATOR.join((uri[:-1] if uri.endswith(':///') else uri for uri in uri_chunks))

    def _params_from_url(self, url, defaults):
        chunks = url.split(self._SERVER_URI_SEPARATOR)
        connparams = dict(defaults, hosts=[])
        for chunk in chunks:
            data = super()._params_from_url(url=chunk, defaults=defaults)
            connparams['hosts'].append(data)
        for param in ('host', 'port', 'db', 'password'):
            connparams.pop(param)
        for param in ('db', 'password'):
            if connparams['hosts'] and param in connparams['hosts'][0]:
                connparams[param] = connparams['hosts'][0].get(param)
        return connparams

    def _get_sentinel_instance(self, **params):
        connparams = params.copy()
        hosts = connparams.pop('hosts')
        min_other_sentinels = self._transport_options.get('min_other_sentinels', 0)
        sentinel_kwargs = self._transport_options.get('sentinel_kwargs', {})
        sentinel_instance = self.sentinel.Sentinel([(cp['host'], cp['port']) for cp in hosts], min_other_sentinels=min_other_sentinels, sentinel_kwargs=sentinel_kwargs, **connparams)
        return sentinel_instance

    def _get_pool(self, **params):
        sentinel_instance = self._get_sentinel_instance(**params)
        master_name = self._transport_options.get('master_name', None)
        return sentinel_instance.master_for(service_name=master_name, redis_class=self._get_client()).connection_pool