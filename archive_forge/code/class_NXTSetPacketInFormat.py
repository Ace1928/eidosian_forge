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
class NXTSetPacketInFormat(NiciraHeader):

    def __init__(self, datapath, packet_in_format):
        super(NXTSetPacketInFormat, self).__init__(datapath, ofproto.NXT_SET_PACKET_IN_FORMAT)
        self.format = packet_in_format

    def _serialize_body(self):
        self.serialize_header()
        msg_pack_into(ofproto.NX_SET_PACKET_IN_FORMAT_PACK_STR, self.buf, ofproto.NICIRA_HEADER_SIZE, self.format)