import logging
from twisted.internet.defer import Deferred, maybeDeferred
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import load_object
def _robots_error(self, failure, netloc):
    if failure.type is not IgnoreRequest:
        key = f'robotstxt/exception_count/{failure.type}'
        self.crawler.stats.inc_value(key)
    rp_dfd = self._parsers[netloc]
    self._parsers[netloc] = None
    rp_dfd.callback(None)