import logging
from twisted.internet.defer import Deferred, maybeDeferred
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import load_object
def _parse_robots(self, response, netloc, spider):
    self.crawler.stats.inc_value('robotstxt/response_count')
    self.crawler.stats.inc_value(f'robotstxt/response_status_count/{response.status}')
    rp = self._parserimpl.from_crawler(self.crawler, response.body)
    rp_dfd = self._parsers[netloc]
    self._parsers[netloc] = rp
    rp_dfd.callback(rp)