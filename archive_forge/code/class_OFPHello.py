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
@_set_msg_type(ofproto.OFPT_HELLO)
class OFPHello(MsgBase):
    """
    Hello message

    When connection is started, the hello message is exchanged between a
    switch and a controller.

    This message is handled by the OSKen framework, so the OSKen application
    do not need to process this typically.

    ========== =========================================================
    Attribute  Description
    ========== =========================================================
    elements   list of ``OFPHelloElemVersionBitmap`` instance
    ========== =========================================================
    """

    def __init__(self, datapath, elements=None):
        elements = elements if elements else []
        super(OFPHello, self).__init__(datapath)
        self.elements = elements

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPHello, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        offset = ofproto.OFP_HELLO_HEADER_SIZE
        elems = []
        while offset < msg.msg_len:
            type_, length = struct.unpack_from(ofproto.OFP_HELLO_ELEM_HEADER_PACK_STR, msg.buf, offset)
            if type_ == ofproto.OFPHET_VERSIONBITMAP:
                elem = OFPHelloElemVersionBitmap.parser(msg.buf, offset)
                elems.append(elem)
            offset += length
        msg.elements = elems
        return msg