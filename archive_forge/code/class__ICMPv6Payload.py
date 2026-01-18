import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
class _ICMPv6Payload(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    """
    Base class for the payload of ICMPv6 packet.
    """