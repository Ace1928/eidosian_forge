from io import BytesIO
from aiokafka.util import WeakMethod
from .abstract import AbstractType
from .types import Schema
def _encode_self(self):
    return self.SCHEMA.encode([self.__dict__[name] for name in self.SCHEMA.names])