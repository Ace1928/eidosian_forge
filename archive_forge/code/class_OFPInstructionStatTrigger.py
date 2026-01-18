import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
@OFPInstruction.register_instruction_type([ofproto.OFPIT_STAT_TRIGGER])
class OFPInstructionStatTrigger(OFPInstruction):
    """
    Statistics triggers instruction

    This instruction defines a set of statistics thresholds using OXS.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Bitmap of the following flags.

                     | OFPSTF_PERIODIC
                     | OFPSTF_ONLY_FIRST
    thresholds       Instance of ``OFPStats``
    ================ ======================================================
    """

    def __init__(self, flags, thresholds, type_=None, len_=None):
        super(OFPInstructionStatTrigger, self).__init__()
        self.type = ofproto.OFPIT_STAT_TRIGGER
        self.len = len_
        self.flags = flags
        self.thresholds = thresholds

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, flags = struct.unpack_from(ofproto.OFP_INSTRUCTION_STAT_TRIGGER_PACK_STR0, buf, offset)
        offset += 8
        thresholds = OFPStats.parser(buf, offset)
        inst = cls(flags, thresholds)
        inst.len = len_
        return inst

    def serialize(self, buf, offset):
        stats_len = self.thresholds.serialize(buf, offset + 8)
        self.len = 8 + stats_len
        msg_pack_into(ofproto.OFP_INSTRUCTION_STAT_TRIGGER_PACK_STR0, buf, offset, self.type, self.len, self.flags)