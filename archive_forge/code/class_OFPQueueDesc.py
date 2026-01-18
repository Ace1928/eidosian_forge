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
class OFPQueueDesc(StringifyMixin):

    def __init__(self, port_no=None, queue_id=None, len_=None, properties=None):
        super(OFPQueueDesc, self).__init__()
        self.port_no = port_no
        self.queue_id = queue_id
        self.len = len_
        self.properties = properties

    @classmethod
    def parser(cls, buf, offset):
        port_no, queue_id, len_ = struct.unpack_from(ofproto.OFP_QUEUE_DESC_PACK_STR, buf, offset)
        props = []
        rest = buf[offset + ofproto.OFP_QUEUE_DESC_SIZE:offset + len_]
        while rest:
            p, rest = OFPQueueDescProp.parse(rest)
            props.append(p)
        ofpqueuedesc = cls(port_no, queue_id, len_, props)
        return ofpqueuedesc