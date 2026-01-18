from __future__ import annotations
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, cast
from twisted.internet.defer import Deferred
from scrapy.crawler import Crawler
from scrapy.dupefilters import BaseDupeFilter
from scrapy.http.request import Request
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.job import job_dir
from scrapy.utils.misc import create_instance, load_object
class BaseSchedulerMeta(type):
    """
    Metaclass to check scheduler classes against the necessary interface
    """

    def __instancecheck__(cls, instance: Any) -> bool:
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass: type) -> bool:
        return hasattr(subclass, 'has_pending_requests') and callable(subclass.has_pending_requests) and hasattr(subclass, 'enqueue_request') and callable(subclass.enqueue_request) and hasattr(subclass, 'next_request') and callable(subclass.next_request)