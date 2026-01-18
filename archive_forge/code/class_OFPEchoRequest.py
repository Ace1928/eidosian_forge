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
@_set_msg_type(ofproto.OFPT_ECHO_REQUEST)
class OFPEchoRequest(MsgBase):
    """
    Echo request message

    This message is handled by the OSKen framework, so the OSKen application
    do not need to process this typically.

    ========== =========================================================
    Attribute  Description
    ========== =========================================================
    data       An arbitrary length data
    ========== =========================================================

    Example::

        def send_echo_request(self, datapath, data):
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPEchoRequest(datapath, data)
            datapath.send_msg(req)

        @set_ev_cls(ofp_event.EventOFPEchoRequest,
                    [HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER])
        def echo_request_handler(self, ev):
            self.logger.debug('OFPEchoRequest received: data=%s',
                              utils.hex_array(ev.msg.data))
    """

    def __init__(self, datapath, data=None):
        super(OFPEchoRequest, self).__init__(datapath)
        self.data = data

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPEchoRequest, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.data = msg.buf[ofproto.OFP_HEADER_SIZE:]
        return msg

    def _serialize_body(self):
        if self.data is not None:
            self.buf += self.data