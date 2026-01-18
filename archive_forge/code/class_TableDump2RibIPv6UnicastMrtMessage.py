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
@TableDump2MrtMessage.register_type(TableDump2MrtRecord.SUBTYPE_RIB_IPV6_UNICAST)
class TableDump2RibIPv6UnicastMrtMessage(TableDump2AfiSafiSpecificRibMrtMessage):
    """
    MRT Message for the TABLE_DUMP_V2 Type and the
    SUBTYPE_RIB_IPV6_MULTICAST subtype.
    """
    _PREFIX_CLS = bgp.IP6AddrPrefix