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
@OFPActionVendor.register_action_vendor(ofproto_common.NX_EXPERIMENTER_ID)
class NXActionHeader(OFPActionVendor):
    _NX_ACTION_SUBTYPES = {}

    @staticmethod
    def register_nx_action_subtype(subtype, len_):

        def _register_nx_action_subtype(cls):
            cls.cls_action_len = len_
            cls.cls_subtype = subtype
            NXActionHeader._NX_ACTION_SUBTYPES[cls.cls_subtype] = cls
            return cls
        return _register_nx_action_subtype

    def __init__(self):
        super(NXActionHeader, self).__init__()
        self.subtype = self.cls_subtype

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_HEADER_PACK_STR, buf, offset, self.type, self.len)

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, vendor, subtype = struct.unpack_from(ofproto.NX_ACTION_HEADER_PACK_STR, buf, offset)
        cls_ = cls._NX_ACTION_SUBTYPES.get(subtype)
        return cls_.parser(buf, offset)