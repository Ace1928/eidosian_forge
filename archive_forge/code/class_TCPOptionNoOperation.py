import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@TCPOption.register(TCP_OPTION_KIND_NO_OPERATION, TCPOption.NO_BODY_OFFSET)
class TCPOptionNoOperation(TCPOption):
    pass