import socket
import sys
import twisted.internet.abstract  # type: ignore
import twisted.internet.asyncioreactor  # type: ignore
from twisted.internet.defer import Deferred  # type: ignore
from twisted.python import failure  # type: ignore
import twisted.names.cache  # type: ignore
import twisted.names.client  # type: ignore
import twisted.names.hosts  # type: ignore
import twisted.names.resolve  # type: ignore
from tornado.concurrent import Future, future_set_exc_info
from tornado.escape import utf8
from tornado import gen
from tornado.netutil import Resolver
import typing
class TwistedResolver(Resolver):
    """Twisted-based asynchronous resolver.

    This is a non-blocking and non-threaded resolver.  It is
    recommended only when threads cannot be used, since it has
    limitations compared to the standard ``getaddrinfo``-based
    `~tornado.netutil.Resolver` and
    `~tornado.netutil.DefaultExecutorResolver`.  Specifically, it returns at
    most one result, and arguments other than ``host`` and ``family``
    are ignored.  It may fail to resolve when ``family`` is not
    ``socket.AF_UNSPEC``.

    Requires Twisted 12.1 or newer.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    .. deprecated:: 6.2
       This class is deprecated and will be removed in Tornado 7.0. Use the default
       thread-based resolver instead.
    """

    def initialize(self) -> None:
        self.reactor = twisted.internet.asyncioreactor.AsyncioSelectorReactor()
        host_resolver = twisted.names.hosts.Resolver('/etc/hosts')
        cache_resolver = twisted.names.cache.CacheResolver(reactor=self.reactor)
        real_resolver = twisted.names.client.Resolver('/etc/resolv.conf', reactor=self.reactor)
        self.resolver = twisted.names.resolve.ResolverChain([host_resolver, cache_resolver, real_resolver])

    @gen.coroutine
    def resolve(self, host: str, port: int, family: int=0) -> 'Generator[Any, Any, List[Tuple[int, Any]]]':
        if twisted.internet.abstract.isIPAddress(host):
            resolved = host
            resolved_family = socket.AF_INET
        elif twisted.internet.abstract.isIPv6Address(host):
            resolved = host
            resolved_family = socket.AF_INET6
        else:
            deferred = self.resolver.getHostByName(utf8(host))
            fut = Future()
            deferred.addBoth(fut.set_result)
            resolved = (yield fut)
            if isinstance(resolved, failure.Failure):
                try:
                    resolved.raiseException()
                except twisted.names.error.DomainError as e:
                    raise IOError(e)
            elif twisted.internet.abstract.isIPAddress(resolved):
                resolved_family = socket.AF_INET
            elif twisted.internet.abstract.isIPv6Address(resolved):
                resolved_family = socket.AF_INET6
            else:
                resolved_family = socket.AF_UNSPEC
        if family != socket.AF_UNSPEC and family != resolved_family:
            raise Exception('Requested socket family %d but got %d' % (family, resolved_family))
        result = [(typing.cast(int, resolved_family), (resolved, port))]
        return result