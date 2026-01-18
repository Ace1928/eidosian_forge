import random
import threading
import time
import typing
from typing import Any
from typing import Mapping
import warnings
from ..api import CacheBackend
from ..api import NO_VALUE
from ... import util
class MemcacheArgs(GenericMemcachedBackend):
    """Mixin which provides support for the 'time' argument to set(),
    'min_compress_len' to other methods.
    """

    def __init__(self, arguments):
        self.min_compress_len = arguments.get('min_compress_len', 0)
        self.set_arguments = {}
        if 'memcached_expire_time' in arguments:
            self.set_arguments['time'] = arguments['memcached_expire_time']
        if 'min_compress_len' in arguments:
            self.set_arguments['min_compress_len'] = arguments['min_compress_len']
        super(MemcacheArgs, self).__init__(arguments)