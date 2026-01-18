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
@OFPAction.register_action_type(ofproto.OFPAT_SET_QUEUE, ofproto.OFP_ACTION_SET_QUEUE_SIZE)
class OFPActionSetQueue(OFPAction):
    """
    Set queue action

    This action sets the queue id that will be used to map a flow to an
    already-configured queue on a port.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    queue_id         Queue ID for the packets
    ================ ======================================================
    """

    def __init__(self, queue_id, type_=None, len_=None):
        super(OFPActionSetQueue, self).__init__()
        self.queue_id = queue_id

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, queue_id = struct.unpack_from(ofproto.OFP_ACTION_SET_QUEUE_PACK_STR, buf, offset)
        return cls(queue_id)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_SET_QUEUE_PACK_STR, buf, offset, self.type, self.len, self.queue_id)