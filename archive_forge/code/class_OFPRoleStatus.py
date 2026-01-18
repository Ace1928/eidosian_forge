import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_4 as ofproto
@_register_parser
@_set_msg_type(ofproto.OFPT_ROLE_STATUS)
class OFPRoleStatus(MsgBase):
    """
    Role status message

    The switch notifies controller of change of role.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    role             One of the following values.

                     | OFPCR_ROLE_NOCHANGE
                     | OFPCR_ROLE_EQUAL
                     | OFPCR_ROLE_MASTER
    reason           One of the following values.

                     | OFPCRR_MASTER_REQUEST
                     | OFPCRR_CONFIG
                     | OFPCRR_EXPERIMENTER
    generation_id    Master Election Generation ID
    properties       List of ``OFPRoleProp`` subclass instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPRoleStatus, MAIN_DISPATCHER)
        def role_status_handler(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto

            if msg.role == ofp.OFPCR_ROLE_NOCHANGE:
                role = 'ROLE NOCHANGE'
            elif msg.role == ofp.OFPCR_ROLE_EQUAL:
                role = 'ROLE EQUAL'
            elif msg.role == ofp.OFPCR_ROLE_MASTER:
                role = 'ROLE MASTER'
            else:
                role = 'unknown'

            if msg.reason == ofp.OFPCRR_MASTER_REQUEST:
                reason = 'MASTER REQUEST'
            elif msg.reason == ofp.OFPCRR_CONFIG:
                reason = 'CONFIG'
            elif msg.reason == ofp.OFPCRR_EXPERIMENTER:
                reason = 'EXPERIMENTER'
            else:
                reason = 'unknown'

            self.logger.debug('OFPRoleStatus received: role=%s reason=%s '
                              'generation_id=%d properties=%s', role, reason,
                              msg.generation_id, repr(msg.properties))
    """

    def __init__(self, datapath, role=None, reason=None, generation_id=None, properties=None):
        super(OFPRoleStatus, self).__init__(datapath)
        self.role = role
        self.reason = reason
        self.generation_id = generation_id
        self.properties = properties

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPRoleStatus, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.role, msg.reason, msg.generation_id = struct.unpack_from(ofproto.OFP_ROLE_STATUS_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)
        msg.properties = []
        rest = msg.buf[ofproto.OFP_ROLE_STATUS_SIZE:]
        while rest:
            p, rest = OFPRoleProp.parse(rest)
            msg.properties.append(p)
        return msg