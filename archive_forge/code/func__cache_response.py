from email.utils import formatdate
from typing import Optional, Type, TypeVar
from twisted.internet import defer
from twisted.internet.error import (
from twisted.web.client import ResponseFailed
from scrapy import signals
from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http.request import Request
from scrapy.http.response import Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.misc import load_object
def _cache_response(self, spider: Spider, response: Response, request: Request, cachedresponse: Optional[Response]) -> None:
    if self.policy.should_cache_response(response, request):
        self.stats.inc_value('httpcache/store', spider=spider)
        self.storage.store_response(spider, request, response)
    else:
        self.stats.inc_value('httpcache/uncacheable', spider=spider)