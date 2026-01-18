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
class OFPControllerStatusStats(StringifyMixin):
    """
    Controller status structure

    ============== =========================================================
    Attribute      Description
    ============== =========================================================
    length         Length of this entry.
    short_id       ID number which identifies the controller.
    role           Bitmap of controller's role flags.

                   | OFPCR_ROLE_NOCHANGE
                   | OFPCR_ROLE_EQUAL
                   | OFPCR_ROLE_MASTER
                   | OFPCR_ROLE_SLAVE
    reason         Bitmap of controller status reason flags.

                   | OFPCSR_REQUEST
                   | OFPCSR_CHANNEL_STATUS
                   | OFPCSR_ROLE
                   | OFPCSR_CONTROLLER_ADDED
                   | OFPCSR_CONTROLLER_REMOVED
                   | OFPCSR_SHORT_ID
                   | OFPCSR_EXPERIMENTER
    channel_status Bitmap of control channel status flags.

                   | OFPCT_STATUS_UP
                   | OFPCT_STATUS_DOWN
    properties     List of ``OFPControllerStatusProp`` subclass instance
    ============== =========================================================
    """

    def __init__(self, short_id=None, role=None, reason=None, channel_status=None, properties=None, length=None):
        super(OFPControllerStatusStats, self).__init__()
        self.length = length
        self.short_id = short_id
        self.role = role
        self.reason = reason
        self.channel_status = channel_status
        self.properties = properties

    @classmethod
    def parser(cls, buf, offset):
        status = cls()
        status.length, status.short_id, status.role, status.reason, status.channel_status = struct.unpack_from(ofproto.OFP_CONTROLLER_STATUS_PACK_STR, buf, offset)
        offset += ofproto.OFP_CONTROLLER_STATUS_SIZE
        status.properties = []
        rest = buf[offset:offset + status.length]
        while rest:
            p, rest = OFPControllerStatusProp.parse(rest)
            status.properties.append(p)
        return status