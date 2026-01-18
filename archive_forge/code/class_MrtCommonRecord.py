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
class MrtCommonRecord(MrtRecord):
    """
    MRT record using MRT Common Header.
    """
    _HEADER_FMT = '!IHHI'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def serialize_header(self):
        return struct.pack(self._HEADER_FMT, self.timestamp, self.type, self.subtype, self.length)