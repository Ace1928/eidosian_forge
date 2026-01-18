import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_sequence_number(MetaSpec):
    type_byte = 0
    attributes = ['number']
    defaults = [0]

    def decode(self, message, data):
        if len(data) == 0:
            message.number = 0
        else:
            message.number = data[0] << 8 | data[1]

    def encode(self, message):
        return [message.number >> 8, message.number & 255]

    def check(self, name, value):
        check_int(value, 0, 65535)