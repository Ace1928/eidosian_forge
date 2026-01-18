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
class Downloader:
    DOWNLOAD_SLOT = 'download_slot'

    def __init__(self, crawler: 'Crawler'):
        self.settings: BaseSettings = crawler.settings
        self.signals: SignalManager = crawler.signals
        self.slots: Dict[str, Slot] = {}
        self.active: Set[Request] = set()
        self.handlers: DownloadHandlers = DownloadHandlers(crawler)
        self.total_concurrency: int = self.settings.getint('CONCURRENT_REQUESTS')
        self.domain_concurrency: int = self.settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN')
        self.ip_concurrency: int = self.settings.getint('CONCURRENT_REQUESTS_PER_IP')
        self.randomize_delay: bool = self.settings.getbool('RANDOMIZE_DOWNLOAD_DELAY')
        self.middleware: DownloaderMiddlewareManager = DownloaderMiddlewareManager.from_crawler(crawler)
        self._slot_gc_loop: task.LoopingCall = task.LoopingCall(self._slot_gc)
        self._slot_gc_loop.start(60)
        self.per_slot_settings: Dict[str, Dict[str, Any]] = self.settings.getdict('DOWNLOAD_SLOTS', {})

    def fetch(self, request: Request, spider: Spider) -> Deferred:

        def _deactivate(response: Response) -> Response:
            self.active.remove(request)
            return response
        self.active.add(request)
        dfd = self.middleware.download(self._enqueue_request, request, spider)
        return dfd.addBoth(_deactivate)

    def needs_backout(self) -> bool:
        return len(self.active) >= self.total_concurrency

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

    def _get_slot_key(self, request: Request, spider: Spider) -> str:
        if self.DOWNLOAD_SLOT in request.meta:
            return cast(str, request.meta[self.DOWNLOAD_SLOT])
        key = urlparse_cached(request).hostname or ''
        if self.ip_concurrency:
            key = dnscache.get(key, key)
        return key

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

    def _process_queue(self, spider: Spider, slot: Slot) -> None:
        from twisted.internet import reactor
        if slot.latercall and slot.latercall.active():
            return
        now = time()
        delay = slot.download_delay()
        if delay:
            penalty = delay - now + slot.lastseen
            if penalty > 0:
                slot.latercall = reactor.callLater(penalty, self._process_queue, spider, slot)
                return
        while slot.queue and slot.free_transfer_slots() > 0:
            slot.lastseen = now
            request, deferred = slot.queue.popleft()
            dfd = self._download(slot, request, spider)
            dfd.chainDeferred(deferred)
            if delay:
                self._process_queue(spider, slot)
                break

    def _download(self, slot: Slot, request: Request, spider: Spider) -> Deferred:
        dfd = mustbe_deferred(self.handlers.download_request, request, spider)

        def _downloaded(response: Response) -> Response:
            self.signals.send_catch_log(signal=signals.response_downloaded, response=response, request=request, spider=spider)
            return response
        dfd.addCallback(_downloaded)
        slot.transferring.add(request)

        def finish_transferring(_: Any) -> Any:
            slot.transferring.remove(request)
            self._process_queue(spider, slot)
            self.signals.send_catch_log(signal=signals.request_left_downloader, request=request, spider=spider)
            return _
        return dfd.addBoth(finish_transferring)

    def close(self) -> None:
        self._slot_gc_loop.stop()
        for slot in self.slots.values():
            slot.close()

    def _slot_gc(self, age: float=60) -> None:
        mintime = time() - age
        for key, slot in list(self.slots.items()):
            if not slot.active and slot.lastseen + slot.delay < mintime:
                self.slots.pop(key).close()