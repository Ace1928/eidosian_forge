import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _register_chunk_type(cls):
    sctp._SCTP_CHUNK_TYPE[cls.chunk_type()] = cls
    return cls