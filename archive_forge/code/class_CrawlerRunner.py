from __future__ import annotations
import logging
import pprint
import signal
import warnings
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional, Set, Type, Union, cast
from twisted.internet.defer import (
from zope.interface.exceptions import DoesNotImplement
from zope.interface.verify import verifyClass
from scrapy import Spider, signals
from scrapy.addons import AddonManager
from scrapy.core.engine import ExecutionEngine
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.extension import ExtensionManager
from scrapy.interfaces import ISpiderLoader
from scrapy.logformatter import LogFormatter
from scrapy.settings import BaseSettings, Settings, overridden_settings
from scrapy.signalmanager import SignalManager
from scrapy.statscollectors import StatsCollector
from scrapy.utils.log import (
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.ossignal import install_shutdown_handlers, signal_names
from scrapy.utils.reactor import (
class CrawlerRunner:
    """
    This is a convenient helper class that keeps track of, manages and runs
    crawlers inside an already setup :mod:`~twisted.internet.reactor`.

    The CrawlerRunner object must be instantiated with a
    :class:`~scrapy.settings.Settings` object.

    This class shouldn't be needed (since Scrapy is responsible of using it
    accordingly) unless writing scripts that manually handle the crawling
    process. See :ref:`run-from-script` for an example.
    """
    crawlers = property(lambda self: self._crawlers, doc='Set of :class:`crawlers <scrapy.crawler.Crawler>` started by :meth:`crawl` and managed by this class.')

    @staticmethod
    def _get_spider_loader(settings: BaseSettings):
        """Get SpiderLoader instance from settings"""
        cls_path = settings.get('SPIDER_LOADER_CLASS')
        loader_cls = load_object(cls_path)
        excs = (DoesNotImplement, MultipleInvalid) if MultipleInvalid else DoesNotImplement
        try:
            verifyClass(ISpiderLoader, loader_cls)
        except excs:
            warnings.warn('SPIDER_LOADER_CLASS (previously named SPIDER_MANAGER_CLASS) does not fully implement scrapy.interfaces.ISpiderLoader interface. Please add all missing methods to avoid unexpected runtime errors.', category=ScrapyDeprecationWarning, stacklevel=2)
        return loader_cls.from_settings(settings.frozencopy())

    def __init__(self, settings: Union[Dict[str, Any], Settings, None]=None):
        if isinstance(settings, dict) or settings is None:
            settings = Settings(settings)
        self.settings = settings
        self.spider_loader = self._get_spider_loader(settings)
        self._crawlers: Set[Crawler] = set()
        self._active: Set[Deferred] = set()
        self.bootstrap_failed = False

    def crawl(self, crawler_or_spidercls: Union[Type[Spider], str, Crawler], *args: Any, **kwargs: Any) -> Deferred:
        """
        Run a crawler with the provided arguments.

        It will call the given Crawler's :meth:`~Crawler.crawl` method, while
        keeping track of it so it can be stopped later.

        If ``crawler_or_spidercls`` isn't a :class:`~scrapy.crawler.Crawler`
        instance, this method will try to create one using this parameter as
        the spider class given to it.

        Returns a deferred that is fired when the crawling is finished.

        :param crawler_or_spidercls: already created crawler, or a spider class
            or spider's name inside the project to create it
        :type crawler_or_spidercls: :class:`~scrapy.crawler.Crawler` instance,
            :class:`~scrapy.spiders.Spider` subclass or string

        :param args: arguments to initialize the spider

        :param kwargs: keyword arguments to initialize the spider
        """
        if isinstance(crawler_or_spidercls, Spider):
            raise ValueError('The crawler_or_spidercls argument cannot be a spider object, it must be a spider class (or a Crawler object)')
        crawler = self.create_crawler(crawler_or_spidercls)
        return self._crawl(crawler, *args, **kwargs)

    def _crawl(self, crawler: Crawler, *args: Any, **kwargs: Any) -> Deferred:
        self.crawlers.add(crawler)
        d = crawler.crawl(*args, **kwargs)
        self._active.add(d)

        def _done(result: Any) -> Any:
            self.crawlers.discard(crawler)
            self._active.discard(d)
            self.bootstrap_failed |= not getattr(crawler, 'spider', None)
            return result
        return d.addBoth(_done)

    def create_crawler(self, crawler_or_spidercls: Union[Type[Spider], str, Crawler]) -> Crawler:
        """
        Return a :class:`~scrapy.crawler.Crawler` object.

        * If ``crawler_or_spidercls`` is a Crawler, it is returned as-is.
        * If ``crawler_or_spidercls`` is a Spider subclass, a new Crawler
          is constructed for it.
        * If ``crawler_or_spidercls`` is a string, this function finds
          a spider with this name in a Scrapy project (using spider loader),
          then creates a Crawler instance for it.
        """
        if isinstance(crawler_or_spidercls, Spider):
            raise ValueError('The crawler_or_spidercls argument cannot be a spider object, it must be a spider class (or a Crawler object)')
        if isinstance(crawler_or_spidercls, Crawler):
            return crawler_or_spidercls
        return self._create_crawler(crawler_or_spidercls)

    def _create_crawler(self, spidercls: Union[str, Type[Spider]]) -> Crawler:
        if isinstance(spidercls, str):
            spidercls = self.spider_loader.load(spidercls)
        return Crawler(cast(Type[Spider], spidercls), self.settings)

    def stop(self) -> Deferred:
        """
        Stops simultaneously all the crawling jobs taking place.

        Returns a deferred that is fired when they all have ended.
        """
        return DeferredList([c.stop() for c in list(self.crawlers)])

    @inlineCallbacks
    def join(self) -> Generator[Deferred, Any, None]:
        """
        join()

        Returns a deferred that is fired when all managed :attr:`crawlers` have
        completed their executions.
        """
        while self._active:
            yield DeferredList(self._active)