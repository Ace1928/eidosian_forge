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
@TableDump2MrtMessage.register_type(TableDump2MrtRecord.SUBTYPE_RIB_GENERIC_ADDPATH)
class TableDump2RibGenericAddPathMrtMessage(TableDump2RibGenericMrtMessage):
    """
    MRT Message for the TABLE_DUMP_V2 Type and the RIB_GENERIC_ADDPATH
    subtype.
    """
    _IS_ADDPATH = True