import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_set_tempo(MetaSpec):
    type_byte = 81
    attributes = ['tempo']
    defaults = [500000]

    def decode(self, message, data):
        message.tempo = data[0] << 16 | data[1] << 8 | data[2]

    def encode(self, message):
        tempo = message.tempo
        return [tempo >> 16, tempo >> 8 & 255, tempo & 255]

    def check(self, name, value):
        check_int(value, 0, 16777215)