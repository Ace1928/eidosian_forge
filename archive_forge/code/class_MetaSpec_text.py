import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_text(MetaSpec):
    type_byte = 1
    attributes = ['text']
    defaults = ['']

    def decode(self, message, data):
        message.text = decode_string(data)

    def encode(self, message):
        return encode_string(message.text)

    def check(self, name, value):
        check_str(value)