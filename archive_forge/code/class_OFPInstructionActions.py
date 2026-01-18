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
@OFPInstruction.register_instruction_type([ofproto.OFPIT_WRITE_ACTIONS, ofproto.OFPIT_APPLY_ACTIONS, ofproto.OFPIT_CLEAR_ACTIONS])
class OFPInstructionActions(OFPInstruction):
    """
    Actions instruction

    This instruction writes/applies/clears the actions.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    type             One of following values.

                     | OFPIT_WRITE_ACTIONS
                     | OFPIT_APPLY_ACTIONS
                     | OFPIT_CLEAR_ACTIONS
    actions          list of OpenFlow action class
    ================ ======================================================

    ``type`` attribute corresponds to ``type_`` parameter of __init__.
    """

    def __init__(self, type_, actions=None, len_=None):
        super(OFPInstructionActions, self).__init__()
        self.type = type_
        for a in actions:
            assert isinstance(a, OFPAction)
        self.actions = actions

    @classmethod
    def parser(cls, buf, offset):
        type_, len_ = struct.unpack_from(ofproto.OFP_INSTRUCTION_ACTIONS_PACK_STR, buf, offset)
        offset += ofproto.OFP_INSTRUCTION_ACTIONS_SIZE
        actions = []
        actions_len = len_ - ofproto.OFP_INSTRUCTION_ACTIONS_SIZE
        exc = None
        try:
            while actions_len > 0:
                a = OFPAction.parser(buf, offset)
                actions.append(a)
                actions_len -= a.len
                offset += a.len
        except struct.error as e:
            exc = e
        inst = cls(type_, actions)
        inst.len = len_
        if exc is not None:
            raise exception.OFPTruncatedMessage(ofpmsg=inst, residue=buf[offset:], original_exception=exc)
        return inst

    def serialize(self, buf, offset):
        action_offset = offset + ofproto.OFP_INSTRUCTION_ACTIONS_SIZE
        if self.actions:
            for a in self.actions:
                a.serialize(buf, action_offset)
                action_offset += a.len
        self.len = action_offset - offset
        pad_len = utils.round_up(self.len, 8) - self.len
        msg_pack_into('%dx' % pad_len, buf, action_offset)
        self.len += pad_len
        msg_pack_into(ofproto.OFP_INSTRUCTION_ACTIONS_PACK_STR, buf, offset, self.type, self.len)