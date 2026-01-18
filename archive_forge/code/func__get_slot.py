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
def _get_slot(self, request: Request, spider: Spider) -> Tuple[str, Slot]:
    key = self._get_slot_key(request, spider)
    if key not in self.slots:
        slot_settings = self.per_slot_settings.get(key, {})
        conc = self.ip_concurrency if self.ip_concurrency else self.domain_concurrency
        conc, delay = _get_concurrency_delay(conc, spider, self.settings)
        conc, delay = (slot_settings.get('concurrency', conc), slot_settings.get('delay', delay))
        randomize_delay = slot_settings.get('randomize_delay', self.randomize_delay)
        new_slot = Slot(conc, delay, randomize_delay)
        self.slots[key] = new_slot
    return (key, self.slots[key])