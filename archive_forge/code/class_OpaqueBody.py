from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
class OpaqueBody(StringifyMixin, type_desc.TypeDisp):

    def __init__(self, tlvs=None):
        tlvs = tlvs if tlvs else []
        self.tlvs = tlvs

    def serialize(self):
        return reduce(lambda a, b: a + b, (tlv.serialize() for tlv in self.tlvs))