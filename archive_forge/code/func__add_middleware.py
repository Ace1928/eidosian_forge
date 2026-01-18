from typing import Any, Callable, Generator, List, Union, cast
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Spider
from scrapy.exceptions import _InvalidOutput
from scrapy.http import Request, Response
from scrapy.middleware import MiddlewareManager
from scrapy.settings import BaseSettings
from scrapy.utils.conf import build_component_list
from scrapy.utils.defer import deferred_from_coro, mustbe_deferred
def _add_middleware(self, mw: Any) -> None:
    if hasattr(mw, 'process_request'):
        self.methods['process_request'].append(mw.process_request)
    if hasattr(mw, 'process_response'):
        self.methods['process_response'].appendleft(mw.process_response)
    if hasattr(mw, 'process_exception'):
        self.methods['process_exception'].appendleft(mw.process_exception)