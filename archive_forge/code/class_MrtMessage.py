import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf
class MrtMessage(stringify.StringifyMixin, type_desc.TypeDisp, metaclass=abc.ABCMeta):
    """
    MRT Message in record.
    """

    @classmethod
    @abc.abstractmethod
    def parse(cls, buf):
        pass

    @abc.abstractmethod
    def serialize(self):
        pass