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
class Slot:
    """Downloader slot"""

    def __init__(self, concurrency: int, delay: float, randomize_delay: bool):
        self.concurrency: int = concurrency
        self.delay: float = delay
        self.randomize_delay: bool = randomize_delay
        self.active: Set[Request] = set()
        self.queue: Deque[Tuple[Request, Deferred]] = deque()
        self.transferring: Set[Request] = set()
        self.lastseen: float = 0
        self.latercall = None

    def free_transfer_slots(self) -> int:
        return self.concurrency - len(self.transferring)

    def download_delay(self) -> float:
        if self.randomize_delay:
            return random.uniform(0.5 * self.delay, 1.5 * self.delay)
        return self.delay

    def close(self) -> None:
        if self.latercall and self.latercall.active():
            self.latercall.cancel()

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f'{cls_name}(concurrency={self.concurrency!r}, delay={self.delay:.2f}, randomize_delay={self.randomize_delay!r})'

    def __str__(self) -> str:
        return f'<downloader.Slot concurrency={self.concurrency!r} delay={self.delay:.2f} randomize_delay={self.randomize_delay!r} len(active)={len(self.active)} len(queue)={len(self.queue)} len(transferring)={len(self.transferring)} lastseen={datetime.fromtimestamp(self.lastseen).isoformat()}>'