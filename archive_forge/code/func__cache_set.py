from __future__ import annotations
import calendar
import logging
import re
import time
from email.utils import parsedate_tz
from typing import TYPE_CHECKING, Collection, Mapping
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.cachecontrol.cache import DictCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.serialize import Serializer
def _cache_set(self, cache_url: str, request: PreparedRequest, response: HTTPResponse, body: bytes | None=None, expires_time: int | None=None) -> None:
    """
        Store the data in the cache.
        """
    if isinstance(self.cache, SeparateBodyBaseCache):
        self.cache.set(cache_url, self.serializer.dumps(request, response, b''), expires=expires_time)
        if body is not None:
            self.cache.set_body(cache_url, body)
    else:
        self.cache.set(cache_url, self.serializer.dumps(request, response, body), expires=expires_time)