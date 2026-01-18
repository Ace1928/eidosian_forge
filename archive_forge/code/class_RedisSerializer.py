import pickle
import random
import re
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
class RedisSerializer:

    def __init__(self, protocol=None):
        self.protocol = pickle.HIGHEST_PROTOCOL if protocol is None else protocol

    def dumps(self, obj):
        if type(obj) is int:
            return obj
        return pickle.dumps(obj, self.protocol)

    def loads(self, data):
        try:
            return int(data)
        except ValueError:
            return pickle.loads(data)