import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class cause(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    _PACK_STR = '!HH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    @abc.abstractmethod
    def cause_code(cls):
        pass

    def __init__(self, length=0):
        self.length = length

    @classmethod
    @abc.abstractmethod
    def parser(cls, buf):
        pass

    def serialize(self):
        if 0 == self.length:
            self.length = self._MIN_LEN
        buf = struct.pack(self._PACK_STR, self.cause_code(), self.length)
        return buf

    def __len__(self):
        length = self.length
        mod = length % 4
        if mod:
            length += 4 - mod
        return length