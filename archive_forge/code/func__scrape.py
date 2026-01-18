from __future__ import annotations
import logging
from collections import deque
from typing import (
from itemadapter import is_item
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Spider, signals
from scrapy.core.spidermw import SpiderMiddlewareManager
from scrapy.exceptions import CloseSpider, DropItem, IgnoreRequest
from scrapy.http import Request, Response
from scrapy.logformatter import LogFormatter
from scrapy.pipelines import ItemPipelineManager
from scrapy.signalmanager import SignalManager
from scrapy.utils.defer import (
from scrapy.utils.log import failure_to_exc_info, logformatter_adapter
from scrapy.utils.misc import load_object, warn_on_generator_with_return_value
from scrapy.utils.spider import iterate_spider_output
def _scrape(self, result: Union[Response, Failure], request: Request, spider: Spider) -> Deferred:
    """
        Handle the downloaded response or failure through the spider callback/errback
        """
    if not isinstance(result, (Response, Failure)):
        raise TypeError(f'Incorrect type: expected Response or Failure, got {type(result)}: {result!r}')
    dfd = self._scrape2(result, request, spider)
    dfd.addErrback(self.handle_spider_error, request, result, spider)
    dfd.addCallback(self.handle_spider_output, request, result, spider)
    return dfd