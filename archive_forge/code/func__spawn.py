import asyncio
import logging
import time
import collections
import sys
import os
import socket
from functools import partial
from .resolver import STSResolver, STSFetchResult
from .constants import QUEUE_LIMIT, CHUNK, REQUEST_LIMIT
from .utils import create_custom_socket, filter_domain, is_ipaddr
from .base_cache import CacheEntry
from . import netstring
def _spawn(reader, writer):

    def done_cb(task, fut):
        self._children.discard(task)
    task = self._loop.create_task(self.handler(reader, writer))
    task.add_done_callback(partial(done_cb, task))
    self._children.add(task)
    self._logger.debug('len(self._children) = %d', len(self._children))