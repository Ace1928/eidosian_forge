import random
from collections import deque
from datetime import datetime
from time import time
from typing import TYPE_CHECKING, Any, Deque, Dict, Set, Tuple, cast
from twisted.internet import task
from twisted.internet.defer import Deferred
from scrapy import Request, Spider, signals
from scrapy.core.downloader.handlers import DownloadHandlers
from scrapy.core.downloader.middleware import DownloaderMiddlewareManager
from scrapy.http import Response
from scrapy.resolver import dnscache
from scrapy.settings import BaseSettings
from scrapy.signalmanager import SignalManager
from scrapy.utils.defer import mustbe_deferred
from scrapy.utils.httpobj import urlparse_cached
def _get_concurrency_delay(concurrency: int, spider: Spider, settings: BaseSettings) -> Tuple[int, float]:
    delay: float = settings.getfloat('DOWNLOAD_DELAY')
    if hasattr(spider, 'download_delay'):
        delay = spider.download_delay
    if hasattr(spider, 'max_concurrent_requests'):
        concurrency = spider.max_concurrent_requests
    return (concurrency, delay)