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
@classmethod
def _get_mwlist_from_settings(cls, settings: BaseSettings) -> List[Any]:
    return build_component_list(settings.getwithbase('DOWNLOADER_MIDDLEWARES'))