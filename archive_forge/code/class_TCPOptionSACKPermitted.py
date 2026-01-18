import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@TCPOption.register(TCP_OPTION_KIND_SACK_PERMITTED, 2)
class TCPOptionSACKPermitted(TCPOption):
    _PACK_STR = '!BB'

    def serialize(self):
        return struct.pack(self._PACK_STR, self.kind, self.length)