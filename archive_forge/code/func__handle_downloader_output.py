import logging
from time import time
from typing import (
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.task import LoopingCall
from twisted.python.failure import Failure
from scrapy import signals
from scrapy.core.downloader import Downloader
from scrapy.core.scraper import Scraper
from scrapy.exceptions import CloseSpider, DontCloseSpider
from scrapy.http import Request, Response
from scrapy.logformatter import LogFormatter
from scrapy.settings import BaseSettings, Settings
from scrapy.signalmanager import SignalManager
from scrapy.spiders import Spider
from scrapy.utils.log import failure_to_exc_info, logformatter_adapter
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.reactor import CallLaterOnce
def _handle_downloader_output(self, result: Union[Request, Response, Failure], request: Request) -> Optional[Deferred]:
    assert self.spider is not None
    if not isinstance(result, (Request, Response, Failure)):
        raise TypeError(f'Incorrect type: expected Request, Response or Failure, got {type(result)}: {result!r}')
    if isinstance(result, Request):
        self.crawl(result)
        return None
    d = self.scraper.enqueue_scrape(result, request, self.spider)
    d.addErrback(lambda f: logger.error('Error while enqueuing downloader output', exc_info=failure_to_exc_info(f), extra={'spider': self.spider}))
    return d