import collections
import itertools
import json
import random
from threading import Lock
from threading import Thread
import time
import uuid
import pytest
from dogpile.cache import CacheRegion
from dogpile.cache import register_backend
from dogpile.cache.api import CacheBackend
from dogpile.cache.api import CacheMutex
from dogpile.cache.api import CantDeserializeException
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import _backend_loader
from .assertions import assert_raises_message
from .assertions import eq_
class MockBackend(CacheBackend):

    def __init__(self, arguments):
        self.arguments = arguments
        self._cache = {}

    def get_mutex(self, key):
        return MockMutex(key)

    def get(self, key):
        try:
            return self._cache[key]
        except KeyError:
            return NO_VALUE

    def get_multi(self, keys):
        return [self.get(key) for key in keys]

    def set(self, key, value):
        self._cache[key] = value

    def set_multi(self, mapping):
        for key, value in mapping.items():
            self.set(key, value)

    def delete(self, key):
        self._cache.pop(key, None)

    def delete_multi(self, keys):
        for key in keys:
            self.delete(key)