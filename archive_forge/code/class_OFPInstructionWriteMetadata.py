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
@OFPInstruction.register_instruction_type([ofproto.OFPIT_WRITE_METADATA])
class OFPInstructionWriteMetadata(OFPInstruction):
    """
    Write metadata instruction

    This instruction writes the masked metadata value into the metadata field.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    metadata         Metadata value to write
    metadata_mask    Metadata write bitmask
    ================ ======================================================
    """

    def __init__(self, metadata, metadata_mask, type_=None, len_=None):
        super(OFPInstructionWriteMetadata, self).__init__()
        self.type = ofproto.OFPIT_WRITE_METADATA
        self.len = ofproto.OFP_INSTRUCTION_WRITE_METADATA_SIZE
        self.metadata = metadata
        self.metadata_mask = metadata_mask

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, metadata, metadata_mask = struct.unpack_from(ofproto.OFP_INSTRUCTION_WRITE_METADATA_PACK_STR, buf, offset)
        return cls(metadata, metadata_mask)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_INSTRUCTION_WRITE_METADATA_PACK_STR, buf, offset, self.type, self.len, self.metadata, self.metadata_mask)