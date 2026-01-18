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
@MrtRecord.register_unknown_type()
class UnknownMrtRecord(MrtCommonRecord):
    """
    MRT record for the UNKNOWN Type.
    """
    MESSAGE_CLS = UnknownMrtMessage