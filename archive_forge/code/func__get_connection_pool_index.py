import pickle
import random
import re
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
def _get_connection_pool_index(self, write):
    if write or len(self._servers) == 1:
        return 0
    return random.randint(1, len(self._servers) - 1)