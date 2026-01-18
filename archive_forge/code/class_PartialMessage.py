import io
import time
from binascii import crc32
from aiokafka.codec import (
from aiokafka.errors import UnsupportedCodecError
from aiokafka.util import WeakMethod
from .struct import Struct
from .types import Int8, Int32, UInt32, Int64, Bytes, Schema, AbstractType
class PartialMessage(bytes):

    def __repr__(self):
        return 'PartialMessage(%s)' % (self,)