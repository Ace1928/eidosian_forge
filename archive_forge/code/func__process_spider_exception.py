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
def _process_spider_exception(self, response: Response, spider: Spider, _failure: Failure, start_index: int=0) -> Union[Failure, MutableChain]:
    exception = _failure.value
    if isinstance(exception, _InvalidOutput):
        return _failure
    method_list = islice(self.methods['process_spider_exception'], start_index, None)
    for method_index, method in enumerate(method_list, start=start_index):
        if method is None:
            continue
        method = cast(Callable, method)
        result = method(response=response, exception=exception, spider=spider)
        if _isiterable(result):
            dfd: Deferred = self._process_spider_output(response, spider, result, method_index + 1)
            if dfd.called:
                return cast(MutableChain, dfd.result)
            msg = f'Async iterable returned from {method.__qualname__} cannot be downgraded'
            raise _InvalidOutput(msg)
        elif result is None:
            continue
        else:
            msg = f'{method.__qualname__} must return None or an iterable, got {type(result)}'
            raise _InvalidOutput(msg)
    return _failure