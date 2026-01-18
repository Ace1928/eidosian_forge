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
@_register_parser
@_set_msg_type(ofproto.OFPT_EXPERIMENTER)
class OFPExperimenter(MsgBase):
    """
    Experimenter extension message

    ============= =========================================================
    Attribute     Description
    ============= =========================================================
    experimenter  Experimenter ID
    exp_type      Experimenter defined
    data          Experimenter defined arbitrary additional data
    ============= =========================================================
    """
    _subtypes = {}

    def __init__(self, datapath, experimenter=None, exp_type=None, data=None):
        super(OFPExperimenter, self).__init__(datapath)
        self.experimenter = experimenter
        self.exp_type = exp_type
        self.data = data

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPExperimenter, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.experimenter, msg.exp_type = struct.unpack_from(ofproto.OFP_EXPERIMENTER_HEADER_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)
        msg.data = msg.buf[ofproto.OFP_EXPERIMENTER_HEADER_SIZE:]
        if (msg.experimenter, msg.exp_type) in cls._subtypes:
            new_msg = cls._subtypes[msg.experimenter, msg.exp_type].parser_subtype(msg)
            new_msg.set_headers(msg.version, msg.msg_type, msg.msg_len, msg.xid)
            new_msg.set_buf(msg.buf)
            return new_msg
        return msg

    def _serialize_body(self):
        assert self.data is not None
        msg_pack_into(ofproto.OFP_EXPERIMENTER_HEADER_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.experimenter, self.exp_type)
        self.buf += self.data