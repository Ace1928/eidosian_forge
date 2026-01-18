from __future__ import annotations
import os
import socket
import threading
from collections import deque
from contextlib import contextmanager
from functools import partial
from itertools import count
from uuid import NAMESPACE_OID, uuid3, uuid4, uuid5
from amqp import ChannelError, RecoverableConnectionError
from .entity import Exchange, Queue
from .log import get_logger
from .serialization import registry as serializers
from .utils.uuid import uuid
def _imaybe_declare(entity, channel, **retry_policy):
    _ensure_channel_is_bound(entity, channel)
    if not entity.channel.connection:
        raise RecoverableConnectionError('channel disconnected')
    return entity.channel.connection.client.ensure(entity, _maybe_declare, **retry_policy)(entity, channel)