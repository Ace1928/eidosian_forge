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
def _ensure_channel_is_bound(entity, channel):
    """Make sure the channel is bound to the entity.

    :param entity: generic kombu nomenclature, generally an exchange or queue
    :param channel: channel to bind to the entity
    :return: the updated entity
    """
    is_bound = entity.is_bound
    if not is_bound:
        if not channel:
            raise ChannelError(f'Cannot bind channel {channel} to entity {entity}')
        entity = entity.bind(channel)
        return entity