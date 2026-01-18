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
@staticmethod
def _get_async_method_pair(mw: Any, methodname: str) -> Union[None, Callable, Tuple[Callable, Callable]]:
    normal_method: Optional[Callable] = getattr(mw, methodname, None)
    methodname_async = methodname + '_async'
    async_method: Optional[Callable] = getattr(mw, methodname_async, None)
    if not async_method:
        return normal_method
    if not normal_method:
        logger.error(f'Middleware {mw.__qualname__} has {methodname_async} without {methodname}, skipping this method.')
        return None
    if not isasyncgenfunction(async_method):
        logger.error(f'{async_method.__qualname__} is not an async generator function, skipping this method.')
        return normal_method
    if isasyncgenfunction(normal_method):
        logger.error(f'{normal_method.__qualname__} is an async generator function while {methodname_async} exists, skipping both methods.')
        return None
    return (normal_method, async_method)