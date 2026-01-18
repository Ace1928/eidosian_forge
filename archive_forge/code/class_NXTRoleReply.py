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
@NiciraHeader.register_nx_subtype(ofproto.NXT_ROLE_REPLY)
class NXTRoleReply(NiciraHeader):

    def __init__(self, datapath, role):
        super(NXTRoleReply, self).__init__(datapath, ofproto.NXT_ROLE_REPLY)
        self.role = role

    @classmethod
    def parser(cls, datapath, buf, offset):
        role, = struct.unpack_from(ofproto.NX_ROLE_PACK_STR, buf, offset)
        return cls(datapath, role)