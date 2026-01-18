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
class OFPActionVendorUnknown(OFPActionVendor):

    def __init__(self, vendor, data=None, type_=None, len_=None):
        super(OFPActionVendorUnknown, self).__init__(vendor=vendor)
        self.data = data

    def serialize(self, buf, offset):
        data = self.data
        if data is None:
            data = bytearray()
        self.len = utils.round_up(len(data), 8) + ofproto.OFP_ACTION_VENDOR_HEADER_SIZE
        super(OFPActionVendorUnknown, self).serialize(buf, offset)
        msg_pack_into('!%ds' % len(self.data), buf, offset + ofproto.OFP_ACTION_VENDOR_HEADER_SIZE, self.data)