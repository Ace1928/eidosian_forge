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
@_set_msg_type(ofproto.OFPT_GROUP_MOD)
class OFPGroupMod(MsgBase):
    """
    Modify group entry message

    The controller sends this message to modify the group table.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    command          One of the following values.

                     | OFPGC_ADD
                     | OFPGC_MODIFY
                     | OFPGC_DELETE
    type             One of the following values.

                     | OFPGT_ALL
                     | OFPGT_SELECT
                     | OFPGT_INDIRECT
                     | OFPGT_FF
    group_id         Group identifier
    buckets          list of ``OFPBucket``
    ================ ======================================================

    ``type`` attribute corresponds to ``type_`` parameter of __init__.

    Example::

        def send_group_mod(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            port = 1
            max_len = 2000
            actions = [ofp_parser.OFPActionOutput(port, max_len)]

            weight = 100
            watch_port = 0
            watch_group = 0
            buckets = [ofp_parser.OFPBucket(weight, watch_port, watch_group,
                                            actions)]

            group_id = 1
            req = ofp_parser.OFPGroupMod(datapath, ofp.OFPGC_ADD,
                                         ofp.OFPGT_SELECT, group_id, buckets)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, command=ofproto.OFPGC_ADD, type_=ofproto.OFPGT_ALL, group_id=0, buckets=None):
        buckets = buckets if buckets else []
        super(OFPGroupMod, self).__init__(datapath)
        self.command = command
        self.type = type_
        self.group_id = group_id
        self.buckets = buckets

    def _serialize_body(self):
        msg_pack_into(ofproto.OFP_GROUP_MOD_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.command, self.type, self.group_id)
        offset = ofproto.OFP_GROUP_MOD_SIZE
        for b in self.buckets:
            b.serialize(self.buf, offset)
            offset += b.len