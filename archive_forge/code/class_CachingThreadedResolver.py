from typing import Any
from twisted.internet import defer
from twisted.internet.base import ThreadedResolver
from twisted.internet.interfaces import (
from zope.interface.declarations import implementer, provider
from scrapy.utils.datatypes import LocalCache
@implementer(IResolverSimple)
class CachingThreadedResolver(ThreadedResolver):
    """
    Default caching resolver. IPv4 only, supports setting a timeout value for DNS requests.
    """

    def __init__(self, reactor, cache_size, timeout):
        super().__init__(reactor)
        dnscache.limit = cache_size
        self.timeout = timeout

    @classmethod
    def from_crawler(cls, crawler, reactor):
        if crawler.settings.getbool('DNSCACHE_ENABLED'):
            cache_size = crawler.settings.getint('DNSCACHE_SIZE')
        else:
            cache_size = 0
        return cls(reactor, cache_size, crawler.settings.getfloat('DNS_TIMEOUT'))

    def install_on_reactor(self):
        self.reactor.installResolver(self)

    def getHostByName(self, name: str, timeout=None):
        if name in dnscache:
            return defer.succeed(dnscache[name])
        timeout = (self.timeout,)
        d = super().getHostByName(name, timeout)
        if dnscache.limit:
            d.addCallback(self._cache_result, name)
        return d

    def _cache_result(self, result, name):
        dnscache[name] = result
        return result