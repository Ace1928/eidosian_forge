import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import stringify
def _register_icmp_type(cls):
    for type_ in args:
        icmp._ICMP_TYPES[type_] = cls
    return cls