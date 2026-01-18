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
def _maybe_declare(entity, channel):
    orig = entity
    _ensure_channel_is_bound(entity, channel)
    if channel is None:
        if not entity.is_bound:
            raise ChannelError(f'channel is None and entity {entity} not bound.')
        channel = entity.channel
    declared = ident = None
    if channel.connection and entity.can_cache_declaration:
        declared = channel.connection.client.declared_entities
        ident = hash(entity)
        if ident in declared:
            return False
    if not channel.connection:
        raise RecoverableConnectionError('channel disconnected')
    entity.declare(channel=channel)
    if declared is not None and ident:
        declared.add(ident)
    if orig is not None:
        orig.name = entity.name
    return True