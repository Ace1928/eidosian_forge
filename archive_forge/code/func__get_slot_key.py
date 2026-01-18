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
def _get_slot_key(self, request: Request, spider: Spider) -> str:
    if self.DOWNLOAD_SLOT in request.meta:
        return cast(str, request.meta[self.DOWNLOAD_SLOT])
    key = urlparse_cached(request).hostname or ''
    if self.ip_concurrency:
        key = dnscache.get(key, key)
    return key