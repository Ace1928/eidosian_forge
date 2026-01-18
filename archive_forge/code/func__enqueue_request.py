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
def _enqueue_request(self, request: Request, spider: Spider) -> Deferred:
    key, slot = self._get_slot(request, spider)
    request.meta[self.DOWNLOAD_SLOT] = key

    def _deactivate(response: Response) -> Response:
        slot.active.remove(request)
        return response
    slot.active.add(request)
    self.signals.send_catch_log(signal=signals.request_reached_downloader, request=request, spider=spider)
    deferred: Deferred = Deferred().addBoth(_deactivate)
    slot.queue.append((request, deferred))
    self._process_queue(spider, slot)
    return deferred