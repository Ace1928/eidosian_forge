import logging
from inspect import isasyncgenfunction, iscoroutine
from itertools import islice
from typing import (
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.exceptions import _InvalidOutput
from scrapy.http import Response
from scrapy.middleware import MiddlewareManager
from scrapy.settings import BaseSettings
from scrapy.utils.asyncgen import as_async_generator, collect_asyncgen
from scrapy.utils.conf import build_component_list
from scrapy.utils.defer import (
from scrapy.utils.python import MutableAsyncChain, MutableChain
def _process_spider_input(self, scrape_func: ScrapeFunc, response: Response, request: Request, spider: Spider) -> Any:
    for method in self.methods['process_spider_input']:
        method = cast(Callable, method)
        try:
            result = method(response=response, spider=spider)
            if result is not None:
                msg = f'{method.__qualname__} must return None or raise an exception, got {type(result)}'
                raise _InvalidOutput(msg)
        except _InvalidOutput:
            raise
        except Exception:
            return scrape_func(Failure(), request, spider)
    return scrape_func(response, request, spider)