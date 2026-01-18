from __future__ import annotations
import datetime
from queue import Empty
import pymongo
from pymongo import MongoClient, errors, uri_parser
from pymongo.cursor import CursorType
from kombu.exceptions import VersionMismatch
from kombu.utils.compat import _detect_environment
from kombu.utils.encoding import bytes_to_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.url import maybe_sanitize_url
from . import virtual
from .base import to_rabbitmq_queue_arguments
def _update_queues_expire(self, queue):
    """Update expiration field on queues documents."""
    expire_at = self._get_queue_expire(queue, 'x-expires')
    if not expire_at:
        return
    self.routing.update_many({'queue': queue}, {'$set': {'expire_at': expire_at}})
    self.queues.update_many({'_id': queue}, {'$set': {'expire_at': expire_at}})