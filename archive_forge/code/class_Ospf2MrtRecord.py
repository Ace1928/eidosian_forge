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
@MrtRecord.register_type(MrtRecord.TYPE_OSPFv2)
class Ospf2MrtRecord(MrtCommonRecord):
    """
    MRT Record for the OSPFv2 Type.
    """
    MESSAGE_CLS = Ospf2MrtMessage

    def __init__(self, message, timestamp=None, type_=None, subtype=0, length=None):
        super(Ospf2MrtRecord, self).__init__(message=message, timestamp=timestamp, type_=type_, subtype=subtype, length=length)