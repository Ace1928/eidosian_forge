import collections
import datetime
import logging
import os
import threading
from typing import (
import grpc
from grpc.experimental import experimental_api
def _evict_locked(self, key: CacheKey):
    channel, _ = self._mapping.pop(key)
    _LOGGER.debug('Evicting channel %s with configuration %s.', channel, key)
    channel.close()
    del channel