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
@TableDumpMrtMessage.register_type(TableDumpMrtRecord.SUBTYPE_AFI_IPv6)
class TableDumpAfiIPv6MrtMessage(TableDumpMrtMessage):
    """
    MRT Message for the TABLE_DUMP Type and the AFI_IPv6 subtype.
    """
    _HEADER_FMT = '!HH16sBBI16sHH'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)