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
def _remove_request(_: Any) -> None:
    assert self.slot
    self.slot.remove_request(request)