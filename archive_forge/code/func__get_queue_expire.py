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
def _get_queue_expire(self, queue, argument):
    """Get expiration header named `argument` of queue definition.

        Note:
        ----
            `queue` must be either queue name or options itself.
        """
    if isinstance(queue, str):
        doc = self.queues.find_one({'_id': queue})
        if not doc:
            return
        data = doc['options']
    else:
        data = queue
    try:
        value = data['arguments'][argument]
    except (KeyError, TypeError):
        return
    return self.get_now() + datetime.timedelta(milliseconds=value)