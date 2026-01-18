from __future__ import annotations
import logging
import sys
import warnings
from logging.config import dictConfig
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union, cast
from twisted.python import log as twisted_log
from twisted.python.failure import Failure
import scrapy
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.settings import Settings
from scrapy.utils.versions import scrapy_components_versions
class LogCounterHandler(logging.Handler):
    """Record log levels count into a crawler stats"""

    def __init__(self, crawler: Crawler, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.crawler: Crawler = crawler

    def emit(self, record: logging.LogRecord) -> None:
        sname = f'log_count/{record.levelname}'
        assert self.crawler.stats
        self.crawler.stats.inc_value(sname)