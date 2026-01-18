from __future__ import annotations
import os
import socket
from collections import defaultdict
from contextlib import contextmanager
from queue import Empty
from kombu.exceptions import ChannelError
from kombu.log import get_logger
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
@contextmanager
def _queue_lock(self, queue):
    """Try to acquire a lock on the Queue.

        It does so by creating a object called 'lock' which is locked by the
        current session..

        This way other nodes are not able to write to the lock object which
        means that they have to wait before the lock is released.

        Arguments:
        ---------
            queue (str): The name of the queue.
        """
    lock = etcd.Lock(self.client, queue)
    lock._uuid = self.lock_value
    logger.debug(f'Acquiring lock {lock.name}')
    lock.acquire(blocking=True, lock_ttl=self.lock_ttl)
    try:
        yield
    finally:
        logger.debug(f'Releasing lock {lock.name}')
        lock.release()