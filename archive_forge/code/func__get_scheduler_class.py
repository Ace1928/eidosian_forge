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
def _get_scheduler_class(self, settings: BaseSettings) -> Type['BaseScheduler']:
    from scrapy.core.scheduler import BaseScheduler
    scheduler_cls: Type = load_object(settings['SCHEDULER'])
    if not issubclass(scheduler_cls, BaseScheduler):
        raise TypeError(f'The provided scheduler class ({settings['SCHEDULER']}) does not fully implement the scheduler interface')
    return scheduler_cls