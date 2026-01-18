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
@OFPInstruction.register_instruction_type([ofproto.OFPIT_METER])
class OFPInstructionMeter(OFPInstruction):
    """
    Meter instruction

    This instruction applies the meter.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    meter_id         Meter instance
    ================ ======================================================
    """

    def __init__(self, meter_id=1, type_=None, len_=None):
        super(OFPInstructionMeter, self).__init__()
        self.type = ofproto.OFPIT_METER
        self.len = ofproto.OFP_INSTRUCTION_METER_SIZE
        self.meter_id = meter_id

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, meter_id = struct.unpack_from(ofproto.OFP_INSTRUCTION_METER_PACK_STR, buf, offset)
        return cls(meter_id)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_INSTRUCTION_METER_PACK_STR, buf, offset, self.type, self.len, self.meter_id)