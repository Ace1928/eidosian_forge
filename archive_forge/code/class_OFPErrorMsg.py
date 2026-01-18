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
@_set_msg_type(ofproto.OFPT_ERROR)
class OFPErrorMsg(MsgBase):
    """
    Error message

    The switch notifies controller of problems by this message.

    ========== =========================================================
    Attribute  Description
    ========== =========================================================
    type       High level type of error
    code       Details depending on the type
    data       Variable length data depending on the type and code
    ========== =========================================================

    ``type`` attribute corresponds to ``type_`` parameter of __init__.

    Types and codes are defined in ``os_ken.ofproto.ofproto``.

    ============================= ===========
    Type                          Code
    ============================= ===========
    OFPET_HELLO_FAILED            OFPHFC_*
    OFPET_BAD_REQUEST             OFPBRC_*
    OFPET_BAD_ACTION              OFPBAC_*
    OFPET_BAD_INSTRUCTION         OFPBIC_*
    OFPET_BAD_MATCH               OFPBMC_*
    OFPET_FLOW_MOD_FAILED         OFPFMFC_*
    OFPET_GROUP_MOD_FAILED        OFPGMFC_*
    OFPET_PORT_MOD_FAILED         OFPPMFC_*
    OFPET_TABLE_MOD_FAILED        OFPTMFC_*
    OFPET_QUEUE_OP_FAILED         OFPQOFC_*
    OFPET_SWITCH_CONFIG_FAILED    OFPSCFC_*
    OFPET_ROLE_REQUEST_FAILED     OFPRRFC_*
    OFPET_METER_MOD_FAILED        OFPMMFC_*
    OFPET_TABLE_FEATURES_FAILED   OFPTFFC_*
    OFPET_EXPERIMENTER            N/A
    ============================= ===========

    If ``type == OFPET_EXPERIMENTER``, this message has also the following
    attributes.

    ============= ======================================================
    Attribute     Description
    ============= ======================================================
    exp_type      Experimenter defined type
    experimenter  Experimenter ID
    ============= ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPErrorMsg,
                    [HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER])
        def error_msg_handler(self, ev):
            msg = ev.msg

            self.logger.debug('OFPErrorMsg received: type=0x%02x code=0x%02x '
                              'message=%s',
                              msg.type, msg.code, utils.hex_array(msg.data))
    """

    def __init__(self, datapath, type_=None, code=None, data=None, **kwargs):
        super(OFPErrorMsg, self).__init__(datapath)
        self.type = type_
        self.code = code
        if isinstance(data, str):
            data = data.encode('ascii')
        self.data = data
        if self.type == ofproto.OFPET_EXPERIMENTER:
            self.exp_type = kwargs.get('exp_type', None)
            self.experimenter = kwargs.get('experimenter', None)

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        type_, = struct.unpack_from('!H', bytes(buf), ofproto.OFP_HEADER_SIZE)
        msg = super(OFPErrorMsg, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        if type_ == ofproto.OFPET_EXPERIMENTER:
            msg.type, msg.exp_type, msg.experimenter, msg.data = cls.parse_experimenter_body(buf)
        else:
            msg.type, msg.code, msg.data = cls.parse_body(buf)
        return msg

    @classmethod
    def parse_body(cls, buf):
        type_, code = struct.unpack_from(ofproto.OFP_ERROR_MSG_PACK_STR, buf, ofproto.OFP_HEADER_SIZE)
        data = buf[ofproto.OFP_ERROR_MSG_SIZE:]
        return (type_, code, data)

    @classmethod
    def parse_experimenter_body(cls, buf):
        type_, exp_type, experimenter = struct.unpack_from(ofproto.OFP_ERROR_EXPERIMENTER_MSG_PACK_STR, buf, ofproto.OFP_HEADER_SIZE)
        data = buf[ofproto.OFP_ERROR_EXPERIMENTER_MSG_SIZE:]
        return (type_, exp_type, experimenter, data)

    def _serialize_body(self):
        assert self.data is not None
        if self.type == ofproto.OFPET_EXPERIMENTER:
            msg_pack_into(ofproto.OFP_ERROR_EXPERIMENTER_MSG_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.type, self.exp_type, self.experimenter)
            self.buf += self.data
        else:
            msg_pack_into(ofproto.OFP_ERROR_MSG_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.type, self.code)
            self.buf += self.data