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
@_register_parser
@_set_msg_type(ofproto.OFPT_CONTROLLER_STATUS)
class OFPControllerStatus(MsgBase):
    """
    Controller status message

    The switch informs the controller about the status of the control
    channel it maintains with each controller.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    status           ``OFPControllerStatusStats`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPControllerStatus, MAIN_DISPATCHER)
        def table(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto
            status = msg.status

            if status.role == ofp.OFPCR_ROLE_NOCHANGE:
                role = 'NOCHANGE'
            elif status.role == ofp.OFPCR_ROLE_EQUAL:
                role = 'EQUAL'
            elif status.role == ofp.OFPCR_ROLE_MASTER:
                role = 'MASTER'
            elif status.role == ofp.OFPCR_ROLE_SLAVE:
                role = 'SLAVE'
            else:
                role = 'unknown'

            if status.reason == ofp.OFPCSR_REQUEST:
                reason = 'REQUEST'
            elif status.reason == ofp.OFPCSR_CHANNEL_STATUS:
                reason = 'CHANNEL_STATUS'
            elif status.reason == ofp.OFPCSR_ROLE:
                reason = 'ROLE'
            elif status.reason == ofp.OFPCSR_CONTROLLER_ADDED:
                reason = 'CONTROLLER_ADDED'
            elif status.reason == ofp.OFPCSR_CONTROLLER_REMOVED:
                reason = 'CONTROLLER_REMOVED'
            elif status.reason == ofp.OFPCSR_SHORT_ID:
                reason = 'SHORT_ID'
            elif status.reason == ofp.OFPCSR_EXPERIMENTER:
                reason = 'EXPERIMENTER'
            else:
                reason = 'unknown'

            if status.channel_status == OFPCT_STATUS_UP:
                channel_status = 'UP'
            if status.channel_status == OFPCT_STATUS_DOWN:
                channel_status = 'DOWN'
            else:
                channel_status = 'unknown'

            self.logger.debug('OFPControllerStatus received: short_id=%d'
                              'role=%s reason=%s channel_status=%s '
                              'properties=%s',
                              status.short_id, role, reason, channel_status,
                              repr(status.properties))
    """

    def __init__(self, datapath, status=None):
        super(OFPControllerStatus, self).__init__(datapath)
        self.status = status

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPControllerStatus, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.status = OFPControllerStatusStats.parser(msg.buf, ofproto.OFP_HEADER_SIZE)
        return msg