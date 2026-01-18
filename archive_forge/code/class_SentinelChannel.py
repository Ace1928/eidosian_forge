from __future__ import annotations
import functools
import numbers
import socket
from bisect import bisect
from collections import namedtuple
from contextlib import contextmanager
from queue import Empty
from time import time
from vine import promise
from kombu.exceptions import InconsistencyError, VersionMismatch
from kombu.log import get_logger
from kombu.utils.compat import register_after_fork
from kombu.utils.encoding import bytes_to_str
from kombu.utils.eventio import ERR, READ, poll
from kombu.utils.functional import accepts_argument
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.scheduling import cycle_by_name
from kombu.utils.url import _parse_url
from . import virtual
class SentinelChannel(Channel):
    """Channel with explicit Redis Sentinel knowledge.

    Broker url is supposed to look like:

    .. code-block::

        sentinel://0.0.0.0:26379;sentinel://0.0.0.0:26380/...

    where each sentinel is separated by a `;`.

    Other arguments for the sentinel should come from the transport options
    (see `transport_options` of :class:`~kombu.connection.Connection`).

    You must provide at least one option in Transport options:
     * `master_name` - name of the redis group to poll

    Example:
    -------
    .. code-block:: python

        >>> import kombu
        >>> c = kombu.Connection(
             'sentinel://sentinel1:26379;sentinel://sentinel2:26379',
             transport_options={'master_name': 'mymaster'}
        )
        >>> c.connect()
    """
    from_transport_options = Channel.from_transport_options + ('master_name', 'min_other_sentinels', 'sentinel_kwargs')
    connection_class = sentinel.SentinelManagedConnection if sentinel else None
    connection_class_ssl = SentinelManagedSSLConnection if sentinel else None

    def _sentinel_managed_pool(self, asynchronous=False):
        connparams = self._connparams(asynchronous)
        additional_params = connparams.copy()
        additional_params.pop('host', None)
        additional_params.pop('port', None)
        sentinels = []
        for url in self.connection.client.alt:
            url = _parse_url(url)
            if url.scheme == 'sentinel':
                port = url.port or self.connection.default_port
                sentinels.append((url.hostname, port))
        if not sentinels:
            sentinels.append((connparams['host'], connparams['port']))
        sentinel_inst = sentinel.Sentinel(sentinels, min_other_sentinels=getattr(self, 'min_other_sentinels', 0), sentinel_kwargs=getattr(self, 'sentinel_kwargs', None), **additional_params)
        master_name = getattr(self, 'master_name', None)
        if master_name is None:
            raise ValueError("'master_name' transport option must be specified.")
        return sentinel_inst.master_for(master_name, redis.Redis).connection_pool

    def _get_pool(self, asynchronous=False):
        return self._sentinel_managed_pool(asynchronous)