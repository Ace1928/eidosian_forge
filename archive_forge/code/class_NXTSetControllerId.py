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
class NXTSetControllerId(NiciraHeader):

    def __init__(self, datapath, controller_id):
        super(NXTSetControllerId, self).__init__(datapath, ofproto.NXT_SET_CONTROLLER_ID)
        self.controller_id = controller_id

    def _serialize_body(self):
        self.serialize_header()
        msg_pack_into(ofproto.NX_CONTROLLER_ID_PACK_STR, self.buf, ofproto.NICIRA_HEADER_SIZE, self.controller_id)