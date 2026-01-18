import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common
class MsgBase(StringifyMixin):
    """
    This is a base class for OpenFlow message classes.

    An instance of this class has at least the following attributes.

    ========= ==============================
    Attribute Description
    ========= ==============================
    datapath  A os_ken.controller.controller.Datapath instance for this message
    version   OpenFlow protocol version
    msg_type  Type of OpenFlow message
    msg_len   Length of the message
    xid       Transaction id
    buf       Raw data
    ========= ==============================
    """

    @create_list_of_base_attributes
    def __init__(self, datapath):
        super(MsgBase, self).__init__()
        self.datapath = datapath
        self.version = None
        self.msg_type = None
        self.msg_len = None
        self.xid = None
        self.buf = None

    def set_headers(self, version, msg_type, msg_len, xid):
        assert msg_type == self.cls_msg_type
        self.version = version
        self.msg_type = msg_type
        self.msg_len = msg_len
        self.xid = xid

    def set_xid(self, xid):
        assert self.xid is None
        self.xid = xid

    def set_buf(self, buf):
        self.buf = bytes(buf)

    def __str__(self):

        def hexify(x):
            return hex(x) if isinstance(x, int) else x
        buf = 'version=%s,msg_type=%s,msg_len=%s,xid=%s,' % (hexify(self.version), hexify(self.msg_type), hexify(self.msg_len), hexify(self.xid))
        return buf + StringifyMixin.__str__(self)

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg_ = cls(datapath)
        msg_.set_headers(version, msg_type, msg_len, xid)
        msg_.set_buf(buf)
        return msg_

    def _serialize_pre(self):
        self.version = self.datapath.ofproto.OFP_VERSION
        self.msg_type = self.cls_msg_type
        self.buf = bytearray(self.datapath.ofproto.OFP_HEADER_SIZE)

    def _serialize_header(self):
        assert self.version is not None
        assert self.msg_type is not None
        assert self.buf is not None
        assert len(self.buf) >= self.datapath.ofproto.OFP_HEADER_SIZE
        self.msg_len = len(self.buf)
        if self.xid is None:
            self.xid = 0
        struct.pack_into(self.datapath.ofproto.OFP_HEADER_PACK_STR, self.buf, 0, self.version, self.msg_type, self.msg_len, self.xid)

    def _serialize_body(self):
        pass

    def serialize(self):
        self._serialize_pre()
        self._serialize_body()
        self._serialize_header()