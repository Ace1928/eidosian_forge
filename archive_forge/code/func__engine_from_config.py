from __future__ import annotations
import threading
from json import dumps, loads
from queue import Empty
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from kombu.transport import virtual
from kombu.utils import cached_property
from kombu.utils.encoding import bytes_to_str
from .models import Message as MessageBase
from .models import ModelBase
from .models import Queue as QueueBase
from .models import class_registry, metadata
def _engine_from_config(self):
    conninfo = self.connection.client
    transport_options = conninfo.transport_options.copy()
    transport_options.pop('queue_tablename', None)
    transport_options.pop('message_tablename', None)
    return create_engine(conninfo.hostname, **transport_options)