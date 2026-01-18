import abc
import logging
from os_ken.lib.packet import packet
class PacketInFilterBase(object, metaclass=abc.ABCMeta):

    def __init__(self, args):
        self.args = args

    @abc.abstractmethod
    def filter(self, pkt):
        pass