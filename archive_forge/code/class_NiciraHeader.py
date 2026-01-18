import struct
import base64
import netaddr
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import mac
from os_ken.lib.packet import packet
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0 as ofproto
from os_ken.ofproto import nx_actions
from os_ken import utils
import logging
@OFPVendor.register_vendor(ofproto_common.NX_EXPERIMENTER_ID)
class NiciraHeader(OFPVendor):
    _NX_SUBTYPES = {}

    @staticmethod
    def register_nx_subtype(subtype):

        def _register_nx_subtype(cls):
            cls.cls_subtype = subtype
            NiciraHeader._NX_SUBTYPES[cls.cls_subtype] = cls
            return cls
        return _register_nx_subtype

    def __init__(self, datapath, subtype):
        super(NiciraHeader, self).__init__(datapath)
        self.vendor = ofproto_common.NX_EXPERIMENTER_ID
        self.subtype = subtype

    def serialize_header(self):
        super(NiciraHeader, self).serialize_header()
        msg_pack_into(ofproto.NICIRA_HEADER_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.vendor, self.subtype)

    @classmethod
    def parser(cls, datapath, buf, offset):
        vendor, subtype = struct.unpack_from(ofproto.NICIRA_HEADER_PACK_STR, buf, offset + ofproto.OFP_HEADER_SIZE)
        cls_ = cls._NX_SUBTYPES.get(subtype)
        return cls_.parser(datapath, buf, offset + ofproto.NICIRA_HEADER_SIZE)