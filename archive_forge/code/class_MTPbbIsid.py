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
@OFPMatchField.register_field_header([ofproto.OXM_OF_PBB_ISID, ofproto.OXM_OF_PBB_ISID_W])
class MTPbbIsid(OFPMatchField):
    pack_str = '!3B'

    def __init__(self, header, value, mask=None):
        super(MTPbbIsid, self).__init__(header)
        self.value = value
        self.mask = mask

    @classmethod
    def field_parser(cls, header, buf, offset):
        mask = None
        if ofproto.oxm_tlv_header_extract_hasmask(header):
            pack_str = '!' + cls.pack_str[1:] * 2
            v1, v2, v3, m1, m2, m3 = struct.unpack_from(pack_str, buf, offset + 4)
            value = v1 << 16 | v2 << 8 | v3
            mask = m1 << 16 | m2 << 8 | m3
        else:
            v1, v2, v3 = struct.unpack_from(cls.pack_str, buf, offset + 4)
            value = v1 << 16 | v2 << 8 | v3
        return cls(header, value, mask)

    def _put(self, buf, offset, value):
        msg_pack_into(self.pack_str, buf, offset, value >> 16 & 255, value >> 8 & 255, value >> 0 & 255)
        self.length += self.n_bytes