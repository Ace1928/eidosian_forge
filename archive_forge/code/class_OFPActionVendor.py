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
@OFPAction.register_action_type(ofproto.OFPAT_VENDOR, 0)
class OFPActionVendor(OFPAction):
    """
    Vendor action

    This action is an extensible action for the vendor.
    """
    _ACTION_VENDORS = {}

    @staticmethod
    def register_action_vendor(vendor):

        def _register_action_vendor(cls):
            cls.cls_vendor = vendor
            OFPActionVendor._ACTION_VENDORS[cls.cls_vendor] = cls
            return cls
        return _register_action_vendor

    def __init__(self, vendor=None):
        super(OFPActionVendor, self).__init__()
        self.type = ofproto.OFPAT_VENDOR
        self.len = None
        if vendor is None:
            self.vendor = self.cls_vendor
        else:
            self.vendor = vendor

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, vendor = struct.unpack_from(ofproto.OFP_ACTION_VENDOR_HEADER_PACK_STR, buf, offset)
        data = buf[offset + ofproto.OFP_ACTION_VENDOR_HEADER_SIZE:offset + len_]
        if vendor == ofproto_common.NX_EXPERIMENTER_ID:
            obj = NXAction.parse(data)
        else:
            cls_ = cls._ACTION_VENDORS.get(vendor, None)
            if cls_ is None:
                obj = OFPActionVendorUnknown(vendor, data)
            else:
                obj = cls_.parser(buf, offset)
        obj.len = len_
        return obj

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_VENDOR_HEADER_PACK_STR, buf, offset, self.type, self.len, self.vendor)