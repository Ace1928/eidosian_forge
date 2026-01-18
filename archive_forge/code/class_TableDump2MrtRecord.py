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
@MrtRecord.register_type(MrtRecord.TYPE_TABLE_DUMP_V2)
class TableDump2MrtRecord(MrtCommonRecord):
    MESSAGE_CLS = TableDump2MrtMessage
    SUBTYPE_PEER_INDEX_TABLE = 1
    SUBTYPE_RIB_IPV4_UNICAST = 2
    SUBTYPE_RIB_IPV4_MULTICAST = 3
    SUBTYPE_RIB_IPV6_UNICAST = 4
    SUBTYPE_RIB_IPV6_MULTICAST = 5
    SUBTYPE_RIB_GENERIC = 6
    SUBTYPE_RIB_IPV4_UNICAST_ADDPATH = 8
    SUBTYPE_RIB_IPV4_MULTICAST_ADDPATH = 9
    SUBTYPE_RIB_IPV6_UNICAST_ADDPATH = 10
    SUBTYPE_RIB_IPV6_MULTICAST_ADDPATH = 11
    SUBTYPE_RIB_GENERIC_ADDPATH = 12