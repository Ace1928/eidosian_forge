import random
import threading
import time
import typing
from typing import Any
from typing import Mapping
import warnings
from ..api import CacheBackend
from ..api import NO_VALUE
from ... import util
class PyMemcacheBackend(GenericMemcachedBackend):
    """A backend for the
    `pymemcache <https://github.com/pinterest/pymemcache>`_
    memcached client.

    A comprehensive, fast, pure Python memcached client

    .. versionadded:: 1.1.2

    pymemcache supports the following features:

    * Complete implementation of the memcached text protocol.
    * Configurable timeouts for socket connect and send/recv calls.
    * Access to the "noreply" flag, which can significantly increase
      the speed of writes.
    * Flexible, simple approach to serialization and deserialization.
    * The (optional) ability to treat network and memcached errors as
      cache misses.

    dogpile.cache uses the ``HashClient`` from pymemcache in order to reduce
    API differences when compared to other memcached client drivers.
    This allows the user to provide a single server or a list of memcached
    servers.

    Arguments which can be passed to the ``arguments``
    dictionary include:

    :param tls_context: optional TLS context, will be used for
     TLS connections.

     A typical configuration using tls_context::

        import ssl
        from dogpile.cache import make_region

        ctx = ssl.create_default_context(cafile="/path/to/my-ca.pem")

        region = make_region().configure(
            'dogpile.cache.pymemcache',
            expiration_time = 3600,
            arguments = {
                'url':["127.0.0.1"],
                'tls_context':ctx,
            }
        )

     .. seealso::

        `<https://docs.python.org/3/library/ssl.html>`_ - additional TLS
        documentation.

    :param serde: optional "serde". Defaults to
     ``pymemcache.serde.pickle_serde``.

    :param default_noreply:  defaults to False.  When set to True this flag
     enables the pymemcache "noreply" feature.  See the pymemcache
     documentation for further details.

    :param socket_keepalive: optional socket keepalive, will be used for
     TCP keepalive configuration.  Use of this parameter requires pymemcache
     3.5.0 or greater.  This parameter
     accepts a
     `pymemcache.client.base.KeepAliveOpts
     <https://pymemcache.readthedocs.io/en/latest/apidoc/pymemcache.client.base.html#pymemcache.client.base.KeepaliveOpts>`_
     object.

     A typical configuration using ``socket_keepalive``::

        from pymemcache import KeepaliveOpts
        from dogpile.cache import make_region

        # Using the default keepalive configuration
        socket_keepalive = KeepaliveOpts()

        region = make_region().configure(
            'dogpile.cache.pymemcache',
            expiration_time = 3600,
            arguments = {
                'url':["127.0.0.1"],
                'socket_keepalive': socket_keepalive
            }
        )

     .. versionadded:: 1.1.4 - added support for ``socket_keepalive``.

    :param enable_retry_client: optional flag to enable retry client
     mechanisms to handle failure.  Defaults to False.  When set to ``True``,
     the :paramref:`.PyMemcacheBackend.retry_attempts` parameter must also
     be set, along with optional parameters
     :paramref:`.PyMemcacheBackend.retry_delay`.
     :paramref:`.PyMemcacheBackend.retry_for`,
     :paramref:`.PyMemcacheBackend.do_not_retry_for`.

     .. seealso::

        `<https://pymemcache.readthedocs.io/en/latest/getting_started.html#using-the-built-in-retrying-mechanism>`_ -
        in the pymemcache documentation

     .. versionadded:: 1.1.4

    :param retry_attempts: how many times to attempt an action with
     pymemcache's retrying wrapper before failing. Must be 1 or above.
     Defaults to None.

     .. versionadded:: 1.1.4

    :param retry_delay: optional int|float, how many seconds to sleep between
     each attempt. Used by the retry wrapper. Defaults to None.

     .. versionadded:: 1.1.4

    :param retry_for: optional None|tuple|set|list, what exceptions to
     allow retries for. Will allow retries for all exceptions if None.
     Example: ``(MemcacheClientError, MemcacheUnexpectedCloseError)``
     Accepts any class that is a subclass of Exception.  Defaults to None.

     .. versionadded:: 1.1.4

    :param do_not_retry_for: optional None|tuple|set|list, what
     exceptions should be retried. Will not block retries for any Exception if
     None. Example: ``(IOError, MemcacheIllegalInputError)``
     Accepts any class that is a subclass of Exception. Defaults to None.

     .. versionadded:: 1.1.4

    :param hashclient_retry_attempts: Amount of times a client should be tried
     before it is marked dead and removed from the pool in the HashClient's
     internal mechanisms.

     .. versionadded:: 1.1.5

    :param hashclient_retry_timeout: Time in seconds that should pass between
     retry attempts in the HashClient's internal mechanisms.

     .. versionadded:: 1.1.5

    :param dead_timeout: Time in seconds before attempting to add a node
     back in the pool in the HashClient's internal mechanisms.

     .. versionadded:: 1.1.5

    """

    def __init__(self, arguments):
        super().__init__(arguments)
        self.serde = arguments.get('serde', pymemcache.serde.pickle_serde)
        self.default_noreply = arguments.get('default_noreply', False)
        self.tls_context = arguments.get('tls_context', None)
        self.socket_keepalive = arguments.get('socket_keepalive', None)
        self.enable_retry_client = arguments.get('enable_retry_client', False)
        self.retry_attempts = arguments.get('retry_attempts', None)
        self.retry_delay = arguments.get('retry_delay', None)
        self.retry_for = arguments.get('retry_for', None)
        self.do_not_retry_for = arguments.get('do_not_retry_for', None)
        self.hashclient_retry_attempts = arguments.get('hashclient_retry_attempts', 2)
        self.hashclient_retry_timeout = arguments.get('hashclient_retry_timeout', 1)
        self.dead_timeout = arguments.get('hashclient_dead_timeout', 60)
        if (self.retry_delay is not None or self.retry_attempts is not None or self.retry_for is not None or (self.do_not_retry_for is not None)) and (not self.enable_retry_client):
            warnings.warn('enable_retry_client is not set; retry options will be ignored')

    def _imports(self):
        global pymemcache
        import pymemcache

    def _create_client(self):
        _kwargs = {'serde': self.serde, 'default_noreply': self.default_noreply, 'tls_context': self.tls_context, 'retry_attempts': self.hashclient_retry_attempts, 'retry_timeout': self.hashclient_retry_timeout, 'dead_timeout': self.dead_timeout}
        if self.socket_keepalive is not None:
            _kwargs.update({'socket_keepalive': self.socket_keepalive})
        client = pymemcache.client.hash.HashClient(self.url, **_kwargs)
        if self.enable_retry_client:
            return pymemcache.client.retrying.RetryingClient(client, attempts=self.retry_attempts, retry_delay=self.retry_delay, retry_for=self.retry_for, do_not_retry_for=self.do_not_retry_for)
        return client