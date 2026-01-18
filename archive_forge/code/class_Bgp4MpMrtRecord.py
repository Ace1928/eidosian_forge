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
@MrtRecord.register_type(MrtRecord.TYPE_BGP4MP)
class Bgp4MpMrtRecord(MrtCommonRecord):
    MESSAGE_CLS = Bgp4MpMrtMessage
    SUBTYPE_BGP4MP_STATE_CHANGE = 0
    SUBTYPE_BGP4MP_MESSAGE = 1
    SUBTYPE_BGP4MP_MESSAGE_AS4 = 4
    SUBTYPE_BGP4MP_STATE_CHANGE_AS4 = 5
    SUBTYPE_BGP4MP_MESSAGE_LOCAL = 6
    SUBTYPE_BGP4MP_MESSAGE_AS4_LOCAL = 7
    SUBTYPE_BGP4MP_MESSAGE_ADDPATH = 8
    SUBTYPE_BGP4MP_MESSAGE_AS4_ADDPATH = 9
    SUBTYPE_BGP4MP_MESSAGE_LOCAL_ADDPATH = 10
    SUBTYPE_BGP4MP_MESSAGE_AS4_LOCAL_ADDPATH = 11