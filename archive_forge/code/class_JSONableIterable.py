import base64
from calendar import timegm
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
import traceback
class JSONableIterable(list):

    def __init__(self, iterable):
        self._iterable = iter(iterable)
        try:
            self._peeked = next(self._iterable)
            self._has_items = True
        except StopIteration:
            self._peeked = None
            self._has_items = False

    def __bool__(self):
        return self._has_items
    __nonzero__ = __bool__

    def __iter__(self):
        if self._has_items:
            yield self._peeked
        for item in self._iterable:
            yield item