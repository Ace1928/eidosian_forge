import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class UnknownMetaMessage(MetaMessage):

    def __init__(self, type_byte, data=None, time=0, type='unknown_meta', **kwargs):
        if data is None:
            data = ()
        else:
            data = tuple(data)
        vars(self).update({'type': type, 'type_byte': type_byte, 'data': data, 'time': time})

    def __repr__(self):
        fmt = 'UnknownMetaMessage(type_byte={}, data={}, time={})'
        return fmt.format(self.type_byte, self.data, self.time)

    def __setattr__(self, name, value):
        vars(self)[name] = value

    def bytes(self):
        length = encode_variable_int(len(self.data))
        return [255, self.type_byte] + length + list(self.data)