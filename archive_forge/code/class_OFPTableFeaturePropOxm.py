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
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_MATCH)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_WILDCARDS)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_WRITE_SETFIELD)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_WRITE_SETFIELD_MISS)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_APPLY_SETFIELD)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_APPLY_SETFIELD_MISS)
class OFPTableFeaturePropOxm(OFPTableFeatureProp):

    def __init__(self, type_=None, length=None, oxm_ids=None):
        oxm_ids = oxm_ids if oxm_ids else []
        super(OFPTableFeaturePropOxm, self).__init__(type_, length)
        self.oxm_ids = oxm_ids

    @classmethod
    def parser(cls, buf):
        rest = cls.get_rest(buf)
        ids = []
        while rest:
            i, rest = OFPOxmId.parse(rest)
            ids.append(i)
        return cls(oxm_ids=ids)

    def serialize_body(self):
        bin_ids = bytearray()
        for i in self.oxm_ids:
            bin_ids += i.serialize()
        return bin_ids