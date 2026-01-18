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
def _declarative_cls(self, name, base, ns):
    if name not in class_registry:
        with _MUTEX:
            if name in class_registry:
                return class_registry[name]
            return type(str(name), (base, ModelBase), ns)
    return class_registry[name]