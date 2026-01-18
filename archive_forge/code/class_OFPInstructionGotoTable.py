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
@OFPInstruction.register_instruction_type([ofproto.OFPIT_GOTO_TABLE])
class OFPInstructionGotoTable(OFPInstruction):
    """
    Goto table instruction

    This instruction indicates the next table in the processing pipeline.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    table_id         Next table
    ================ ======================================================
    """

    def __init__(self, table_id, type_=None, len_=None):
        super(OFPInstructionGotoTable, self).__init__()
        self.type = ofproto.OFPIT_GOTO_TABLE
        self.len = ofproto.OFP_INSTRUCTION_GOTO_TABLE_SIZE
        self.table_id = table_id

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, table_id = struct.unpack_from(ofproto.OFP_INSTRUCTION_GOTO_TABLE_PACK_STR, buf, offset)
        return cls(table_id)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_INSTRUCTION_GOTO_TABLE_PACK_STR, buf, offset, self.type, self.len, self.table_id)