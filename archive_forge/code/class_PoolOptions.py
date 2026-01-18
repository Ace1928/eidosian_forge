from __future__ import annotations
import collections
import contextlib
import copy
import os
import platform
import socket
import ssl
import sys
import threading
import time
import weakref
from typing import (
import bson
from bson import DEFAULT_CODEC_OPTIONS
from bson.son import SON
from pymongo import __version__, _csot, auth, helpers
from pymongo.client_session import _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.lock import _create_lock
from pymongo.monitoring import (
from pymongo.network import command, receive_message
from pymongo.read_preferences import ReadPreference
from pymongo.server_api import _add_to_command
from pymongo.server_type import SERVER_TYPE
from pymongo.socket_checker import SocketChecker
from pymongo.ssl_support import HAS_SNI, SSLError
class PoolOptions:
    """Read only connection pool options for a MongoClient.

    Should not be instantiated directly by application developers. Access
    a client's pool options via
    :attr:`~pymongo.client_options.ClientOptions.pool_options` instead::

      pool_opts = client.options.pool_options
      pool_opts.max_pool_size
      pool_opts.min_pool_size

    """
    __slots__ = ('__max_pool_size', '__min_pool_size', '__max_idle_time_seconds', '__connect_timeout', '__socket_timeout', '__wait_queue_timeout', '__ssl_context', '__tls_allow_invalid_hostnames', '__event_listeners', '__appname', '__driver', '__metadata', '__compression_settings', '__max_connecting', '__pause_enabled', '__server_api', '__load_balanced', '__credentials')

    def __init__(self, max_pool_size: int=MAX_POOL_SIZE, min_pool_size: int=MIN_POOL_SIZE, max_idle_time_seconds: Optional[int]=MAX_IDLE_TIME_SEC, connect_timeout: Optional[float]=None, socket_timeout: Optional[float]=None, wait_queue_timeout: Optional[int]=WAIT_QUEUE_TIMEOUT, ssl_context: Optional[SSLContext]=None, tls_allow_invalid_hostnames: bool=False, event_listeners: Optional[_EventListeners]=None, appname: Optional[str]=None, driver: Optional[DriverInfo]=None, compression_settings: Optional[CompressionSettings]=None, max_connecting: int=MAX_CONNECTING, pause_enabled: bool=True, server_api: Optional[ServerApi]=None, load_balanced: Optional[bool]=None, credentials: Optional[MongoCredential]=None):
        self.__max_pool_size = max_pool_size
        self.__min_pool_size = min_pool_size
        self.__max_idle_time_seconds = max_idle_time_seconds
        self.__connect_timeout = connect_timeout
        self.__socket_timeout = socket_timeout
        self.__wait_queue_timeout = wait_queue_timeout
        self.__ssl_context = ssl_context
        self.__tls_allow_invalid_hostnames = tls_allow_invalid_hostnames
        self.__event_listeners = event_listeners
        self.__appname = appname
        self.__driver = driver
        self.__compression_settings = compression_settings
        self.__max_connecting = max_connecting
        self.__pause_enabled = pause_enabled
        self.__server_api = server_api
        self.__load_balanced = load_balanced
        self.__credentials = credentials
        self.__metadata = copy.deepcopy(_METADATA)
        if appname:
            self.__metadata['application'] = {'name': appname}
        if driver:
            if driver.name:
                self.__metadata['driver']['name'] = '{}|{}'.format(_METADATA['driver']['name'], driver.name)
            if driver.version:
                self.__metadata['driver']['version'] = '{}|{}'.format(_METADATA['driver']['version'], driver.version)
            if driver.platform:
                self.__metadata['platform'] = '{}|{}'.format(_METADATA['platform'], driver.platform)
        env = _metadata_env()
        if env:
            self.__metadata['env'] = env
        _truncate_metadata(self.__metadata)

    @property
    def _credentials(self) -> Optional[MongoCredential]:
        """A :class:`~pymongo.auth.MongoCredentials` instance or None."""
        return self.__credentials

    @property
    def non_default_options(self) -> dict[str, Any]:
        """The non-default options this pool was created with.

        Added for CMAP's :class:`PoolCreatedEvent`.
        """
        opts = {}
        if self.__max_pool_size != MAX_POOL_SIZE:
            opts['maxPoolSize'] = self.__max_pool_size
        if self.__min_pool_size != MIN_POOL_SIZE:
            opts['minPoolSize'] = self.__min_pool_size
        if self.__max_idle_time_seconds != MAX_IDLE_TIME_SEC:
            assert self.__max_idle_time_seconds is not None
            opts['maxIdleTimeMS'] = self.__max_idle_time_seconds * 1000
        if self.__wait_queue_timeout != WAIT_QUEUE_TIMEOUT:
            assert self.__wait_queue_timeout is not None
            opts['waitQueueTimeoutMS'] = self.__wait_queue_timeout * 1000
        if self.__max_connecting != MAX_CONNECTING:
            opts['maxConnecting'] = self.__max_connecting
        return opts

    @property
    def max_pool_size(self) -> float:
        """The maximum allowable number of concurrent connections to each
        connected server. Requests to a server will block if there are
        `maxPoolSize` outstanding connections to the requested server.
        Defaults to 100. Cannot be 0.

        When a server's pool has reached `max_pool_size`, operations for that
        server block waiting for a socket to be returned to the pool. If
        ``waitQueueTimeoutMS`` is set, a blocked operation will raise
        :exc:`~pymongo.errors.ConnectionFailure` after a timeout.
        By default ``waitQueueTimeoutMS`` is not set.
        """
        return self.__max_pool_size

    @property
    def min_pool_size(self) -> int:
        """The minimum required number of concurrent connections that the pool
        will maintain to each connected server. Default is 0.
        """
        return self.__min_pool_size

    @property
    def max_connecting(self) -> int:
        """The maximum number of concurrent connection creation attempts per
        pool. Defaults to 2.
        """
        return self.__max_connecting

    @property
    def pause_enabled(self) -> bool:
        return self.__pause_enabled

    @property
    def max_idle_time_seconds(self) -> Optional[int]:
        """The maximum number of seconds that a connection can remain
        idle in the pool before being removed and replaced. Defaults to
        `None` (no limit).
        """
        return self.__max_idle_time_seconds

    @property
    def connect_timeout(self) -> Optional[float]:
        """How long a connection can take to be opened before timing out."""
        return self.__connect_timeout

    @property
    def socket_timeout(self) -> Optional[float]:
        """How long a send or receive on a socket can take before timing out."""
        return self.__socket_timeout

    @property
    def wait_queue_timeout(self) -> Optional[int]:
        """How long a thread will wait for a socket from the pool if the pool
        has no free sockets.
        """
        return self.__wait_queue_timeout

    @property
    def _ssl_context(self) -> Optional[SSLContext]:
        """An SSLContext instance or None."""
        return self.__ssl_context

    @property
    def tls_allow_invalid_hostnames(self) -> bool:
        """If True skip ssl.match_hostname."""
        return self.__tls_allow_invalid_hostnames

    @property
    def _event_listeners(self) -> Optional[_EventListeners]:
        """An instance of pymongo.monitoring._EventListeners."""
        return self.__event_listeners

    @property
    def appname(self) -> Optional[str]:
        """The application name, for sending with hello in server handshake."""
        return self.__appname

    @property
    def driver(self) -> Optional[DriverInfo]:
        """Driver name and version, for sending with hello in handshake."""
        return self.__driver

    @property
    def _compression_settings(self) -> Optional[CompressionSettings]:
        return self.__compression_settings

    @property
    def metadata(self) -> SON[str, Any]:
        """A dict of metadata about the application, driver, os, and platform."""
        return self.__metadata.copy()

    @property
    def server_api(self) -> Optional[ServerApi]:
        """A pymongo.server_api.ServerApi or None."""
        return self.__server_api

    @property
    def load_balanced(self) -> Optional[bool]:
        """True if this Pool is configured in load balanced mode."""
        return self.__load_balanced