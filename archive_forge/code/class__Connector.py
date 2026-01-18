import functools
import socket
import numbers
import datetime
import ssl
import typing
from tornado.concurrent import Future, future_add_done_callback
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado import gen
from tornado.netutil import Resolver
from tornado.gen import TimeoutError
from typing import Any, Union, Dict, Tuple, List, Callable, Iterator, Optional
class _Connector(object):
    """A stateless implementation of the "Happy Eyeballs" algorithm.

    "Happy Eyeballs" is documented in RFC6555 as the recommended practice
    for when both IPv4 and IPv6 addresses are available.

    In this implementation, we partition the addresses by family, and
    make the first connection attempt to whichever address was
    returned first by ``getaddrinfo``.  If that connection fails or
    times out, we begin a connection in parallel to the first address
    of the other family.  If there are additional failures we retry
    with other addresses, keeping one connection attempt per family
    in flight at a time.

    http://tools.ietf.org/html/rfc6555

    """

    def __init__(self, addrinfo: List[Tuple], connect: Callable[[socket.AddressFamily, Tuple], Tuple[IOStream, 'Future[IOStream]']]) -> None:
        self.io_loop = IOLoop.current()
        self.connect = connect
        self.future = Future()
        self.timeout = None
        self.connect_timeout = None
        self.last_error = None
        self.remaining = len(addrinfo)
        self.primary_addrs, self.secondary_addrs = self.split(addrinfo)
        self.streams = set()

    @staticmethod
    def split(addrinfo: List[Tuple]) -> Tuple[List[Tuple[socket.AddressFamily, Tuple]], List[Tuple[socket.AddressFamily, Tuple]]]:
        """Partition the ``addrinfo`` list by address family.

        Returns two lists.  The first list contains the first entry from
        ``addrinfo`` and all others with the same family, and the
        second list contains all other addresses (normally one list will
        be AF_INET and the other AF_INET6, although non-standard resolvers
        may return additional families).
        """
        primary = []
        secondary = []
        primary_af = addrinfo[0][0]
        for af, addr in addrinfo:
            if af == primary_af:
                primary.append((af, addr))
            else:
                secondary.append((af, addr))
        return (primary, secondary)

    def start(self, timeout: float=_INITIAL_CONNECT_TIMEOUT, connect_timeout: Optional[Union[float, datetime.timedelta]]=None) -> 'Future[Tuple[socket.AddressFamily, Any, IOStream]]':
        self.try_connect(iter(self.primary_addrs))
        self.set_timeout(timeout)
        if connect_timeout is not None:
            self.set_connect_timeout(connect_timeout)
        return self.future

    def try_connect(self, addrs: Iterator[Tuple[socket.AddressFamily, Tuple]]) -> None:
        try:
            af, addr = next(addrs)
        except StopIteration:
            if self.remaining == 0 and (not self.future.done()):
                self.future.set_exception(self.last_error or IOError('connection failed'))
            return
        stream, future = self.connect(af, addr)
        self.streams.add(stream)
        future_add_done_callback(future, functools.partial(self.on_connect_done, addrs, af, addr))

    def on_connect_done(self, addrs: Iterator[Tuple[socket.AddressFamily, Tuple]], af: socket.AddressFamily, addr: Tuple, future: 'Future[IOStream]') -> None:
        self.remaining -= 1
        try:
            stream = future.result()
        except Exception as e:
            if self.future.done():
                return
            self.last_error = e
            self.try_connect(addrs)
            if self.timeout is not None:
                self.io_loop.remove_timeout(self.timeout)
                self.on_timeout()
            return
        self.clear_timeouts()
        if self.future.done():
            stream.close()
        else:
            self.streams.discard(stream)
            self.future.set_result((af, addr, stream))
            self.close_streams()

    def set_timeout(self, timeout: float) -> None:
        self.timeout = self.io_loop.add_timeout(self.io_loop.time() + timeout, self.on_timeout)

    def on_timeout(self) -> None:
        self.timeout = None
        if not self.future.done():
            self.try_connect(iter(self.secondary_addrs))

    def clear_timeout(self) -> None:
        if self.timeout is not None:
            self.io_loop.remove_timeout(self.timeout)

    def set_connect_timeout(self, connect_timeout: Union[float, datetime.timedelta]) -> None:
        self.connect_timeout = self.io_loop.add_timeout(connect_timeout, self.on_connect_timeout)

    def on_connect_timeout(self) -> None:
        if not self.future.done():
            self.future.set_exception(TimeoutError())
        self.close_streams()

    def clear_timeouts(self) -> None:
        if self.timeout is not None:
            self.io_loop.remove_timeout(self.timeout)
        if self.connect_timeout is not None:
            self.io_loop.remove_timeout(self.connect_timeout)

    def close_streams(self) -> None:
        for stream in self.streams:
            stream.close()