import struct
import base64
import netaddr
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import mac
from os_ken.lib.packet import packet
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0 as ofproto
from os_ken.ofproto import nx_actions
from os_ken import utils
import logging
@_register_parser
@_set_msg_type(ofproto.OFPT_STATS_REPLY)
class OFPStatsReply(MsgBase):
    _STATS_MSG_TYPES = {}

    @staticmethod
    def register_stats_type(body_single_struct=False):

        def _register_stats_type(cls):
            assert cls.cls_stats_type is not None
            assert cls.cls_stats_type not in OFPStatsReply._STATS_MSG_TYPES
            assert cls.cls_stats_body_cls is not None
            cls.cls_body_single_struct = body_single_struct
            OFPStatsReply._STATS_MSG_TYPES[cls.cls_stats_type] = cls
            return cls
        return _register_stats_type

    def __init__(self, datapath):
        super(OFPStatsReply, self).__init__(datapath)
        self.type = None
        self.flags = None
        self.body = None

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
        msg.body = msg.parser_stats_body(msg.buf, msg.msg_len, ofproto.OFP_STATS_MSG_SIZE)
        return msg

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        type_, flags = struct.unpack_from(ofproto.OFP_STATS_MSG_PACK_STR, bytes(buf), ofproto.OFP_HEADER_SIZE)
        stats_type_cls = cls._STATS_MSG_TYPES.get(type_)
        msg = stats_type_cls.parser_stats(datapath, version, msg_type, msg_len, xid, buf)
        msg.type = type_
        msg.flags = flags
        return msg