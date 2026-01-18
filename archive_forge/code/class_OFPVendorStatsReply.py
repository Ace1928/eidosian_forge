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
@OFPStatsReply.register_stats_type()
@_set_stats_type(ofproto.OFPST_VENDOR, OFPVendorStats)
@_set_msg_type(ofproto.OFPT_STATS_REPLY)
class OFPVendorStatsReply(OFPStatsReply):
    """
    Vendor statistics reply message

    The switch responds with a stats reply that include this message to
    an vendor statistics request.
    """
    _STATS_VENDORS = {}

    @staticmethod
    def register_stats_vendor(vendor):

        def _register_stats_vendor(cls):
            cls.cls_vendor = vendor
            OFPVendorStatsReply._STATS_VENDORS[cls.cls_vendor] = cls
            return cls
        return _register_stats_vendor

    def __init__(self, datapath):
        super(OFPVendorStatsReply, self).__init__(datapath)

    @classmethod
    def parser_stats(cls, datapath, version, msg_type, msg_len, xid, buf):
        type_, = struct.unpack_from(ofproto.OFP_VENDOR_STATS_MSG_PACK_STR, bytes(buf), ofproto.OFP_STATS_MSG_SIZE)
        cls_ = cls._STATS_VENDORS.get(type_)
        if cls_ is None:
            msg = MsgBase.parser.__func__(cls, datapath, version, msg_type, msg_len, xid, buf)
            body_cls = cls.cls_stats_body_cls
            body = body_cls.parser(buf, ofproto.OFP_STATS_MSG_SIZE)
            msg.body = body
            return msg
        return cls_.parser(datapath, version, msg_type, msg_len, xid, buf, ofproto.OFP_VENDOR_STATS_MSG_SIZE)