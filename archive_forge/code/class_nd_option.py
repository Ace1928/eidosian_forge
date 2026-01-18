import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
class nd_option(stringify.StringifyMixin, metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def option_type(cls):
        pass

    @abc.abstractmethod
    def __init__(self, _type, length):
        self._type = _type
        self.length = length

    @classmethod
    @abc.abstractmethod
    def parser(cls, buf):
        pass

    @abc.abstractmethod
    def serialize(self):
        pass

    def __len__(self):
        return self._MIN_LEN