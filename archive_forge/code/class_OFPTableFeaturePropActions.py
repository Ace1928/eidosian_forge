import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_WRITE_ACTIONS)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_WRITE_ACTIONS_MISS)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_APPLY_ACTIONS)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_APPLY_ACTIONS_MISS)
class OFPTableFeaturePropActions(OFPTableFeatureProp):

    def __init__(self, type_=None, length=None, action_ids=None):
        action_ids = action_ids if action_ids else []
        super(OFPTableFeaturePropActions, self).__init__(type_, length)
        self.action_ids = action_ids

    @classmethod
    def parser(cls, buf):
        rest = cls.get_rest(buf)
        ids = []
        while rest:
            i, rest = OFPActionId.parse(rest)
            ids.append(i)
        return cls(action_ids=ids)

    def serialize_body(self):
        bin_ids = bytearray()
        for i in self.action_ids:
            bin_ids += i.serialize()
        return bin_ids