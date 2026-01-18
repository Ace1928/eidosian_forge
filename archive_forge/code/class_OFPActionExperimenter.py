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
@OFPAction.register_action_type(ofproto.OFPAT_EXPERIMENTER, ofproto.OFP_ACTION_EXPERIMENTER_HEADER_SIZE)
class OFPActionExperimenter(OFPAction):
    """
    Experimenter action

    This action is an extensible action for the experimenter.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    experimenter     Experimenter ID
    ================ ======================================================

    .. Note::

        For the list of the supported Nicira experimenter actions,
        please refer to :ref:`os_ken.ofproto.nx_actions <nx_actions_structures>`.
    """

    def __init__(self, experimenter):
        super(OFPActionExperimenter, self).__init__()
        self.type = ofproto.OFPAT_EXPERIMENTER
        self.experimenter = experimenter
        self.len = None

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, experimenter = struct.unpack_from(ofproto.OFP_ACTION_EXPERIMENTER_HEADER_PACK_STR, buf, offset)
        data = buf[offset + ofproto.OFP_ACTION_EXPERIMENTER_HEADER_SIZE:offset + len_]
        if experimenter == ofproto_common.NX_EXPERIMENTER_ID:
            obj = NXAction.parse(data)
        else:
            obj = OFPActionExperimenterUnknown(experimenter, data)
        obj.len = len_
        return obj

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_EXPERIMENTER_HEADER_PACK_STR, buf, offset, self.type, self.len, self.experimenter)