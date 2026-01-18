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
class OFPOxmId(StringifyMixin):
    _PACK_STR = '!I'
    _EXPERIMENTER_ID_PACK_STR = '!I'
    _TYPE = {'ascii': ['type']}

    def __init__(self, type_, hasmask=False, length=None):
        self.type = type_
        self.hasmask = hasmask
        self.length = length

    @classmethod
    def parse(cls, buf):
        oxm, = struct.unpack_from(cls._PACK_STR, bytes(buf), 0)
        type_, _v = ofproto.oxm_to_user(oxm >> 1 + 8, None, None)
        rest = buf[struct.calcsize(cls._PACK_STR):]
        hasmask = ofproto.oxm_tlv_header_extract_hasmask(oxm)
        length = oxm & 255
        class_ = oxm >> 7 + 1 + 8
        if class_ == ofproto.OFPXMC_EXPERIMENTER:
            exp_id, = struct.unpack_from(cls._EXPERIMENTER_ID_PACK_STR, bytes(rest), 0)
            rest = rest[struct.calcsize(cls._EXPERIMENTER_ID_PACK_STR):]
            subcls = OFPExperimenterOxmId
            return (subcls(type_=type_, exp_id=exp_id, hasmask=hasmask, length=length), rest)
        else:
            return (cls(type_=type_, hasmask=hasmask, length=length), rest)

    def serialize(self):
        self.length = 0
        n, _v, _m = ofproto.oxm_from_user(self.type, None)
        oxm = n << 1 + 8 | self.hasmask << 8 | self.length
        buf = bytearray()
        msg_pack_into(self._PACK_STR, buf, 0, oxm)
        assert n >> 7 != ofproto.OFPXMC_EXPERIMENTER
        return buf