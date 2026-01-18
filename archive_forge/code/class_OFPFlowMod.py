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
@_set_msg_type(ofproto.OFPT_FLOW_MOD)
class OFPFlowMod(MsgBase):
    """
    Modify Flow entry message

    The controller sends this message to modify the flow table.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    cookie           Opaque controller-issued identifier
    cookie_mask      Mask used to restrict the cookie bits that must match
                     when the command is ``OPFFC_MODIFY*`` or
                     ``OFPFC_DELETE*``
    table_id         ID of the table to put the flow in
    command          One of the following values.

                     | OFPFC_ADD
                     | OFPFC_MODIFY
                     | OFPFC_MODIFY_STRICT
                     | OFPFC_DELETE
                     | OFPFC_DELETE_STRICT
    idle_timeout     Idle time before discarding (seconds)
    hard_timeout     Max time before discarding (seconds)
    priority         Priority level of flow entry
    buffer_id        Buffered packet to apply to (or OFP_NO_BUFFER)
    out_port         For ``OFPFC_DELETE*`` commands, require matching
                     entries to include this as an output port
    out_group        For ``OFPFC_DELETE*`` commands, require matching
                     entries to include this as an output group
    flags            Bitmap of the following flags.

                     | OFPFF_SEND_FLOW_REM
                     | OFPFF_CHECK_OVERLAP
                     | OFPFF_RESET_COUNTS
                     | OFPFF_NO_PKT_COUNTS
                     | OFPFF_NO_BYT_COUNTS
    match            Instance of ``OFPMatch``
    instructions     list of ``OFPInstruction*`` instance
    ================ ======================================================

    Example::

        def send_flow_mod(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            cookie = cookie_mask = 0
            table_id = 0
            idle_timeout = hard_timeout = 0
            priority = 32768
            buffer_id = ofp.OFP_NO_BUFFER
            match = ofp_parser.OFPMatch(in_port=1, eth_dst='ff:ff:ff:ff:ff:ff')
            actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
            inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                     actions)]
            req = ofp_parser.OFPFlowMod(datapath, cookie, cookie_mask,
                                        table_id, ofp.OFPFC_ADD,
                                        idle_timeout, hard_timeout,
                                        priority, buffer_id,
                                        ofp.OFPP_ANY, ofp.OFPG_ANY,
                                        ofp.OFPFF_SEND_FLOW_REM,
                                        match, inst)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, cookie=0, cookie_mask=0, table_id=0, command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0, priority=ofproto.OFP_DEFAULT_PRIORITY, buffer_id=ofproto.OFP_NO_BUFFER, out_port=0, out_group=0, flags=0, match=None, instructions=None):
        instructions = instructions if instructions else []
        super(OFPFlowMod, self).__init__(datapath)
        self.cookie = cookie
        self.cookie_mask = cookie_mask
        self.table_id = table_id
        self.command = command
        self.idle_timeout = idle_timeout
        self.hard_timeout = hard_timeout
        self.priority = priority
        self.buffer_id = buffer_id
        self.out_port = out_port
        self.out_group = out_group
        self.flags = flags
        if match is None:
            match = OFPMatch()
        assert isinstance(match, OFPMatch)
        self.match = match
        for i in instructions:
            assert isinstance(i, OFPInstruction)
        self.instructions = instructions

    def _serialize_body(self):
        msg_pack_into(ofproto.OFP_FLOW_MOD_PACK_STR0, self.buf, ofproto.OFP_HEADER_SIZE, self.cookie, self.cookie_mask, self.table_id, self.command, self.idle_timeout, self.hard_timeout, self.priority, self.buffer_id, self.out_port, self.out_group, self.flags)
        offset = ofproto.OFP_FLOW_MOD_SIZE - ofproto.OFP_MATCH_SIZE
        match_len = self.match.serialize(self.buf, offset)
        offset += match_len
        for inst in self.instructions:
            inst.serialize(self.buf, offset)
            offset += inst.len

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPFlowMod, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.cookie, msg.cookie_mask, msg.table_id, msg.command, msg.idle_timeout, msg.hard_timeout, msg.priority, msg.buffer_id, msg.out_port, msg.out_group, msg.flags = struct.unpack_from(ofproto.OFP_FLOW_MOD_PACK_STR0, msg.buf, ofproto.OFP_HEADER_SIZE)
        offset = ofproto.OFP_FLOW_MOD_SIZE - ofproto.OFP_HEADER_SIZE
        try:
            msg.match = OFPMatch.parser(buf, offset)
        except exception.OFPTruncatedMessage as e:
            msg.match = e.ofpmsg
            e.ofpmsg = msg
            raise e
        offset += utils.round_up(msg.match.length, 8)
        instructions = []
        try:
            while offset < msg_len:
                i = OFPInstruction.parser(buf, offset)
                if i is not None:
                    instructions.append(i)
                    offset += i.len
        except exception.OFPTruncatedMessage as e:
            instructions.append(e.ofpmsg)
            msg.instructions = instructions
            e.ofpmsg = msg
            raise e
        except struct.error as e:
            msg.instructions = instructions
            raise exception.OFPTruncatedMessage(ofpmsg=msg, residue=buf[offset:], original_exception=e)
        msg.instructions = instructions
        return msg