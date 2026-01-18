import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
class MemoryCache(object):
    """httplib2 Cache implementation which only caches locally."""

    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

    def delete(self, key):
        self.cache.pop(key, None)