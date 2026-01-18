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
@_set_msg_type(ofproto.OFPT_REQUESTFORWARD)
class OFPRequestForward(MsgInMsgBase):
    """
    Forwarded request message

    The swtich forwards request messages from one controller to other
    controllers.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    request          ``OFPGroupMod`` or ``OFPMeterMod`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPRequestForward, MAIN_DISPATCHER)
        def request_forward_handler(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto

            if msg.request.msg_type == ofp.OFPT_GROUP_MOD:
                self.logger.debug(
                    'OFPRequestForward received: request=OFPGroupMod('
                    'command=%d, type=%d, group_id=%d, buckets=%s)',
                    msg.request.command, msg.request.type,
                    msg.request.group_id, msg.request.buckets)
            elif msg.request.msg_type == ofp.OFPT_METER_MOD:
                self.logger.debug(
                    'OFPRequestForward received: request=OFPMeterMod('
                    'command=%d, flags=%d, meter_id=%d, bands=%s)',
                    msg.request.command, msg.request.flags,
                    msg.request.meter_id, msg.request.bands)
            else:
                self.logger.debug(
                    'OFPRequestForward received: request=Unknown')
    """

    def __init__(self, datapath, request=None):
        super(OFPRequestForward, self).__init__(datapath)
        self.request = request

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPRequestForward, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        req_buf = buf[ofproto.OFP_HEADER_SIZE:]
        _ver, _type, _len, _xid = ofproto_parser.header(req_buf)
        msg.request = ofproto_parser.msg(datapath, _ver, _type, _len, _xid, req_buf)
        return msg

    def _serialize_body(self):
        assert isinstance(self.request, (OFPGroupMod, OFPMeterMod))
        self.request.serialize()
        self.buf += self.request.buf