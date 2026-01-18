import asyncio
import functools
import random
import sys
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import suppress
from http import HTTPStatus
from http.cookies import SimpleCookie
from itertools import cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
import attr
from . import hdrs, helpers
from .abc import AbstractResolver
from .client_exceptions import (
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import ceil_timeout, get_running_loop, is_ip_address, noop, sentinel
from .locks import EventResultOrError
from .resolver import DefaultResolver
class BaseConnector:
    """Base connector class.

    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    enable_cleanup_closed - Enables clean-up closed ssl transports.
                            Disabled by default.
    timeout_ceil_threshold - Trigger ceiling of timeout values when
                             it's above timeout_ceil_threshold.
    loop - Optional event loop.
    """
    _closed = True
    _source_traceback = None
    _cleanup_closed_period = 2.0

    def __init__(self, *, keepalive_timeout: Union[object, None, float]=sentinel, force_close: bool=False, limit: int=100, limit_per_host: int=0, enable_cleanup_closed: bool=False, loop: Optional[asyncio.AbstractEventLoop]=None, timeout_ceil_threshold: float=5) -> None:
        if force_close:
            if keepalive_timeout is not None and keepalive_timeout is not sentinel:
                raise ValueError('keepalive_timeout cannot be set if force_close is True')
        elif keepalive_timeout is sentinel:
            keepalive_timeout = 15.0
        loop = get_running_loop(loop)
        self._timeout_ceil_threshold = timeout_ceil_threshold
        self._closed = False
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))
        self._conns: Dict[ConnectionKey, List[Tuple[ResponseHandler, float]]] = {}
        self._limit = limit
        self._limit_per_host = limit_per_host
        self._acquired: Set[ResponseHandler] = set()
        self._acquired_per_host: DefaultDict[ConnectionKey, Set[ResponseHandler]] = defaultdict(set)
        self._keepalive_timeout = cast(float, keepalive_timeout)
        self._force_close = force_close
        self._waiters = defaultdict(deque)
        self._loop = loop
        self._factory = functools.partial(ResponseHandler, loop=loop)
        self.cookies = SimpleCookie()
        self._cleanup_handle: Optional[asyncio.TimerHandle] = None
        self._cleanup_closed_handle: Optional[asyncio.TimerHandle] = None
        self._cleanup_closed_disabled = not enable_cleanup_closed
        self._cleanup_closed_transports: List[Optional[asyncio.Transport]] = []
        self._cleanup_closed()

    def __del__(self, _warnings: Any=warnings) -> None:
        if self._closed:
            return
        if not self._conns:
            return
        conns = [repr(c) for c in self._conns.values()]
        self._close()
        kwargs = {'source': self}
        _warnings.warn(f'Unclosed connector {self!r}', ResourceWarning, **kwargs)
        context = {'connector': self, 'connections': conns, 'message': 'Unclosed connector'}
        if self._source_traceback is not None:
            context['source_traceback'] = self._source_traceback
        self._loop.call_exception_handler(context)

    def __enter__(self) -> 'BaseConnector':
        warnings.warn('"with Connector():" is deprecated, use "async with Connector():" instead', DeprecationWarning)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._close()

    async def __aenter__(self) -> 'BaseConnector':
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]]=None, exc_value: Optional[BaseException]=None, exc_traceback: Optional[TracebackType]=None) -> None:
        await self.close()

    @property
    def force_close(self) -> bool:
        """Ultimately close connection on releasing if True."""
        return self._force_close

    @property
    def limit(self) -> int:
        """The total number for simultaneous connections.

        If limit is 0 the connector has no limit.
        The default limit size is 100.
        """
        return self._limit

    @property
    def limit_per_host(self) -> int:
        """The limit for simultaneous connections to the same endpoint.

        Endpoints are the same if they are have equal
        (host, port, is_ssl) triple.
        """
        return self._limit_per_host

    def _cleanup(self) -> None:
        """Cleanup unused transports."""
        if self._cleanup_handle:
            self._cleanup_handle.cancel()
            self._cleanup_handle = None
        now = self._loop.time()
        timeout = self._keepalive_timeout
        if self._conns:
            connections = {}
            deadline = now - timeout
            for key, conns in self._conns.items():
                alive = []
                for proto, use_time in conns:
                    if proto.is_connected():
                        if use_time - deadline < 0:
                            transport = proto.transport
                            proto.close()
                            if key.is_ssl and (not self._cleanup_closed_disabled):
                                self._cleanup_closed_transports.append(transport)
                        else:
                            alive.append((proto, use_time))
                    else:
                        transport = proto.transport
                        proto.close()
                        if key.is_ssl and (not self._cleanup_closed_disabled):
                            self._cleanup_closed_transports.append(transport)
                if alive:
                    connections[key] = alive
            self._conns = connections
        if self._conns:
            self._cleanup_handle = helpers.weakref_handle(self, '_cleanup', timeout, self._loop, timeout_ceil_threshold=self._timeout_ceil_threshold)

    def _drop_acquired_per_host(self, key: 'ConnectionKey', val: ResponseHandler) -> None:
        acquired_per_host = self._acquired_per_host
        if key not in acquired_per_host:
            return
        conns = acquired_per_host[key]
        conns.remove(val)
        if not conns:
            del self._acquired_per_host[key]

    def _cleanup_closed(self) -> None:
        """Double confirmation for transport close.

        Some broken ssl servers may leave socket open without proper close.
        """
        if self._cleanup_closed_handle:
            self._cleanup_closed_handle.cancel()
        for transport in self._cleanup_closed_transports:
            if transport is not None:
                transport.abort()
        self._cleanup_closed_transports = []
        if not self._cleanup_closed_disabled:
            self._cleanup_closed_handle = helpers.weakref_handle(self, '_cleanup_closed', self._cleanup_closed_period, self._loop, timeout_ceil_threshold=self._timeout_ceil_threshold)

    def close(self) -> Awaitable[None]:
        """Close all opened transports."""
        self._close()
        return _DeprecationWaiter(noop())

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._loop.is_closed():
                return
            if self._cleanup_handle:
                self._cleanup_handle.cancel()
            if self._cleanup_closed_handle:
                self._cleanup_closed_handle.cancel()
            for data in self._conns.values():
                for proto, t0 in data:
                    proto.close()
            for proto in self._acquired:
                proto.close()
            for transport in self._cleanup_closed_transports:
                if transport is not None:
                    transport.abort()
        finally:
            self._conns.clear()
            self._acquired.clear()
            self._waiters.clear()
            self._cleanup_handle = None
            self._cleanup_closed_transports.clear()
            self._cleanup_closed_handle = None

    @property
    def closed(self) -> bool:
        """Is connector closed.

        A readonly property.
        """
        return self._closed

    def _available_connections(self, key: 'ConnectionKey') -> int:
        """
        Return number of available connections.

        The limit, limit_per_host and the connection key are taken into account.

        If it returns less than 1 means that there are no connections
        available.
        """
        if self._limit:
            available = self._limit - len(self._acquired)
            if self._limit_per_host and available > 0 and (key in self._acquired_per_host):
                acquired = self._acquired_per_host.get(key)
                assert acquired is not None
                available = self._limit_per_host - len(acquired)
        elif self._limit_per_host and key in self._acquired_per_host:
            acquired = self._acquired_per_host.get(key)
            assert acquired is not None
            available = self._limit_per_host - len(acquired)
        else:
            available = 1
        return available

    async def connect(self, req: ClientRequest, traces: List['Trace'], timeout: 'ClientTimeout') -> Connection:
        """Get from pool or create new connection."""
        key = req.connection_key
        available = self._available_connections(key)
        if available <= 0 or key in self._waiters:
            fut = self._loop.create_future()
            self._waiters[key].append(fut)
            if traces:
                for trace in traces:
                    await trace.send_connection_queued_start()
            try:
                await fut
            except BaseException as e:
                if key in self._waiters:
                    try:
                        self._waiters[key].remove(fut)
                    except ValueError:
                        pass
                raise e
            finally:
                if key in self._waiters and (not self._waiters[key]):
                    del self._waiters[key]
            if traces:
                for trace in traces:
                    await trace.send_connection_queued_end()
        proto = self._get(key)
        if proto is None:
            placeholder = cast(ResponseHandler, _TransportPlaceholder())
            self._acquired.add(placeholder)
            self._acquired_per_host[key].add(placeholder)
            if traces:
                for trace in traces:
                    await trace.send_connection_create_start()
            try:
                proto = await self._create_connection(req, traces, timeout)
                if self._closed:
                    proto.close()
                    raise ClientConnectionError('Connector is closed.')
            except BaseException:
                if not self._closed:
                    self._acquired.remove(placeholder)
                    self._drop_acquired_per_host(key, placeholder)
                    self._release_waiter()
                raise
            else:
                if not self._closed:
                    self._acquired.remove(placeholder)
                    self._drop_acquired_per_host(key, placeholder)
            if traces:
                for trace in traces:
                    await trace.send_connection_create_end()
        elif traces:
            placeholder = cast(ResponseHandler, _TransportPlaceholder())
            self._acquired.add(placeholder)
            self._acquired_per_host[key].add(placeholder)
            for trace in traces:
                await trace.send_connection_reuseconn()
            self._acquired.remove(placeholder)
            self._drop_acquired_per_host(key, placeholder)
        self._acquired.add(proto)
        self._acquired_per_host[key].add(proto)
        return Connection(self, key, proto, self._loop)

    def _get(self, key: 'ConnectionKey') -> Optional[ResponseHandler]:
        try:
            conns = self._conns[key]
        except KeyError:
            return None
        t1 = self._loop.time()
        while conns:
            proto, t0 = conns.pop()
            if proto.is_connected():
                if t1 - t0 > self._keepalive_timeout:
                    transport = proto.transport
                    proto.close()
                    if key.is_ssl and (not self._cleanup_closed_disabled):
                        self._cleanup_closed_transports.append(transport)
                else:
                    if not conns:
                        del self._conns[key]
                    return proto
            else:
                transport = proto.transport
                proto.close()
                if key.is_ssl and (not self._cleanup_closed_disabled):
                    self._cleanup_closed_transports.append(transport)
        del self._conns[key]
        return None

    def _release_waiter(self) -> None:
        """
        Iterates over all waiters until one to be released is found.

        The one to be released is not finished and
        belongs to a host that has available connections.
        """
        if not self._waiters:
            return
        queues = list(self._waiters.keys())
        random.shuffle(queues)
        for key in queues:
            if self._available_connections(key) < 1:
                continue
            waiters = self._waiters[key]
            while waiters:
                waiter = waiters.popleft()
                if not waiter.done():
                    waiter.set_result(None)
                    return

    def _release_acquired(self, key: 'ConnectionKey', proto: ResponseHandler) -> None:
        if self._closed:
            return
        try:
            self._acquired.remove(proto)
            self._drop_acquired_per_host(key, proto)
        except KeyError:
            pass
        else:
            self._release_waiter()

    def _release(self, key: 'ConnectionKey', protocol: ResponseHandler, *, should_close: bool=False) -> None:
        if self._closed:
            return
        self._release_acquired(key, protocol)
        if self._force_close:
            should_close = True
        if should_close or protocol.should_close:
            transport = protocol.transport
            protocol.close()
            if key.is_ssl and (not self._cleanup_closed_disabled):
                self._cleanup_closed_transports.append(transport)
        else:
            conns = self._conns.get(key)
            if conns is None:
                conns = self._conns[key] = []
            conns.append((protocol, self._loop.time()))
            if self._cleanup_handle is None:
                self._cleanup_handle = helpers.weakref_handle(self, '_cleanup', self._keepalive_timeout, self._loop, timeout_ceil_threshold=self._timeout_ceil_threshold)

    async def _create_connection(self, req: ClientRequest, traces: List['Trace'], timeout: 'ClientTimeout') -> ResponseHandler:
        raise NotImplementedError()