import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class param(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    _PACK_STR = '!HH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    @abc.abstractmethod
    def param_type(cls):
        pass

    def __init__(self, value=None, length=0):
        self.length = length
        self.value = value

    @classmethod
    def parser(cls, buf):
        _, length = struct.unpack_from(cls._PACK_STR, buf)
        value = None
        if cls._MIN_LEN < length:
            fmt = '%ds' % (length - cls._MIN_LEN)
            value, = struct.unpack_from(fmt, buf, cls._MIN_LEN)
        return cls(value, length)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.param_type(), self.length))
        if self.value:
            buf.extend(self.value)
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        mod = len(buf) % 4
        if mod:
            buf.extend(bytearray(4 - mod))
        return bytes(buf)

    def __len__(self):
        length = self.length
        mod = length % 4
        if mod:
            length += 4 - mod
        return length