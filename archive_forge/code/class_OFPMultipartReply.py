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
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPMultipartReply(MsgBase):
    _STATS_MSG_TYPES = {}

    @staticmethod
    def register_stats_type(body_single_struct=False):

        def _register_stats_type(cls):
            assert cls.cls_stats_type is not None
            assert cls.cls_stats_type not in OFPMultipartReply._STATS_MSG_TYPES
            assert cls.cls_stats_body_cls is not None
            cls.cls_body_single_struct = body_single_struct
            OFPMultipartReply._STATS_MSG_TYPES[cls.cls_stats_type] = cls
            return cls
        return _register_stats_type

    def __init__(self, datapath, body=None, flags=None):
        super(OFPMultipartReply, self).__init__(datapath)
        self.body = body
        self.flags = flags

    @classmethod
    def parser_stats_body(cls, buf, msg_len, offset):
        body_cls = cls.cls_stats_body_cls
        body = []
        while offset < msg_len:
            entry = body_cls.parser(buf, offset)
            body.append(entry)
            offset += entry.length
        if cls.cls_body_single_struct:
            return body[0]
        return body

    @classmethod
    def parser_stats(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = MsgBase.parser.__func__(cls, datapath, version, msg_type, msg_len, xid, buf)
        msg.body = msg.parser_stats_body(msg.buf, msg.msg_len, ofproto.OFP_MULTIPART_REPLY_SIZE)
        return msg

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        type_, flags = struct.unpack_from(ofproto.OFP_MULTIPART_REPLY_PACK_STR, bytes(buf), ofproto.OFP_HEADER_SIZE)
        stats_type_cls = cls._STATS_MSG_TYPES.get(type_)
        msg = super(OFPMultipartReply, stats_type_cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.type = type_
        msg.flags = flags
        if stats_type_cls is not None:
            offset = ofproto.OFP_MULTIPART_REPLY_SIZE
            body = []
            while offset < msg_len:
                b = stats_type_cls.cls_stats_body_cls.parser(msg.buf, offset)
                body.append(b)
                offset += b.length if hasattr(b, 'length') else b.len
            if stats_type_cls.cls_body_single_struct:
                msg.body = body[0]
            else:
                msg.body = body
        return msg