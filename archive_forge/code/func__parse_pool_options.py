from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, cast
from bson.codec_options import _parse_codec_options
from pymongo import common
from pymongo.auth import MongoCredential, _build_credentials_tuple
from pymongo.common import validate_boolean
from pymongo.compression_support import CompressionSettings
from pymongo.errors import ConfigurationError
from pymongo.monitoring import _EventListener, _EventListeners
from pymongo.pool import PoolOptions
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import (
from pymongo.server_selectors import any_server_selector
from pymongo.ssl_support import get_ssl_context
from pymongo.write_concern import WriteConcern
def _parse_pool_options(username: str, password: str, database: Optional[str], options: Mapping[str, Any]) -> PoolOptions:
    """Parse connection pool options."""
    credentials = _parse_credentials(username, password, database, options)
    max_pool_size = options.get('maxpoolsize', common.MAX_POOL_SIZE)
    min_pool_size = options.get('minpoolsize', common.MIN_POOL_SIZE)
    max_idle_time_seconds = options.get('maxidletimems', common.MAX_IDLE_TIME_SEC)
    if max_pool_size is not None and min_pool_size > max_pool_size:
        raise ValueError('minPoolSize must be smaller or equal to maxPoolSize')
    connect_timeout = options.get('connecttimeoutms', common.CONNECT_TIMEOUT)
    socket_timeout = options.get('sockettimeoutms')
    wait_queue_timeout = options.get('waitqueuetimeoutms', common.WAIT_QUEUE_TIMEOUT)
    event_listeners = cast(Optional[Sequence[_EventListener]], options.get('event_listeners'))
    appname = options.get('appname')
    driver = options.get('driver')
    server_api = options.get('server_api')
    compression_settings = CompressionSettings(options.get('compressors', []), options.get('zlibcompressionlevel', -1))
    ssl_context, tls_allow_invalid_hostnames = _parse_ssl_options(options)
    load_balanced = options.get('loadbalanced')
    max_connecting = options.get('maxconnecting', common.MAX_CONNECTING)
    return PoolOptions(max_pool_size, min_pool_size, max_idle_time_seconds, connect_timeout, socket_timeout, wait_queue_timeout, ssl_context, tls_allow_invalid_hostnames, _EventListeners(event_listeners), appname, driver, compression_settings, max_connecting=max_connecting, server_api=server_api, load_balanced=load_balanced, credentials=credentials)