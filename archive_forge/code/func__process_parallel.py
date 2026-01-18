from __future__ import annotations
import logging
import pprint
from collections import defaultdict, deque
from typing import (
from twisted.internet.defer import Deferred
from scrapy import Spider
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.defer import process_chain, process_parallel
from scrapy.utils.misc import create_instance, load_object
def _process_parallel(self, methodname: str, obj: Any, *args: Any) -> Deferred:
    methods = cast(Iterable[Callable], self.methods[methodname])
    return process_parallel(methods, obj, *args)