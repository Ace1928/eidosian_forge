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
class OFPActionDlAddr(OFPAction):

    def __init__(self, dl_addr):
        super(OFPActionDlAddr, self).__init__()
        if isinstance(dl_addr, (str, str)) and netaddr.valid_mac(dl_addr):
            dl_addr = addrconv.mac.text_to_bin(dl_addr)
        self.dl_addr = dl_addr

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, dl_addr = struct.unpack_from(ofproto.OFP_ACTION_DL_ADDR_PACK_STR, buf, offset)
        assert type_ in (ofproto.OFPAT_SET_DL_SRC, ofproto.OFPAT_SET_DL_DST)
        assert len_ == ofproto.OFP_ACTION_DL_ADDR_SIZE
        return cls(dl_addr)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_DL_ADDR_PACK_STR, buf, offset, self.type, self.len, self.dl_addr)

    def to_jsondict(self):
        body = {'dl_addr': addrconv.mac.bin_to_text(self.dl_addr)}
        return {self.__class__.__name__: body}

    @classmethod
    def from_jsondict(cls, dict_):
        return cls(**dict_)