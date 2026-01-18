import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_time_signature(MetaSpec):
    type_byte = 88
    attributes = ['numerator', 'denominator', 'clocks_per_click', 'notated_32nd_notes_per_beat']
    defaults = [4, 4, 24, 8]

    def decode(self, message, data):
        message.numerator = data[0]
        message.denominator = 2 ** data[1]
        message.clocks_per_click = data[2]
        message.notated_32nd_notes_per_beat = data[3]

    def encode(self, message):
        return [message.numerator, int(math.log(message.denominator, 2)), message.clocks_per_click, message.notated_32nd_notes_per_beat]

    def check(self, name, value):
        if name == 'denominator':
            check_int(value, 1, 2 ** 255)
            encoded = math.log(value, 2)
            encoded_int = int(encoded)
            if encoded != encoded_int:
                raise ValueError('denominator must be a power of 2')
        else:
            check_int(value, 0, 255)