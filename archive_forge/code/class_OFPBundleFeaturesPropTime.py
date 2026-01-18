import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
@OFPBundleFeaturesProp.register_type(ofproto.OFPTMPBF_TIME_CAPABILITY)
class OFPBundleFeaturesPropTime(OFPBundleFeaturesProp):

    def __init__(self, type_=None, length=None, sched_accuracy=None, sched_max_future=None, sched_max_past=None, timestamp=None):
        super(OFPBundleFeaturesPropTime, self).__init__(type_, length)
        self.sched_accuracy = sched_accuracy
        self.sched_max_future = sched_max_future
        self.sched_max_past = sched_max_past
        self.timestamp = timestamp

    @classmethod
    def parser(cls, buf):
        prop = cls()
        prop.type, prop.length = struct.unpack_from(ofproto.OFP_BUNDLE_FEATURES_PROP_TIME_0_PACK_STR, buf)
        offset = ofproto.OFP_BUNDLE_FEATURES_PROP_TIME_0_SIZE
        for f in ['sched_accuracy', 'sched_max_future', 'sched_max_past', 'timestamp']:
            t = OFPTime.parser(buf, offset)
            setattr(prop, f, t)
            offset += ofproto.OFP_TIME_SIZE
        return prop

    def serialize(self):
        self.length = ofproto.OFP_BUNDLE_FEATURES_PROP_TIME_SIZE
        buf = bytearray()
        msg_pack_into(ofproto.OFP_BUNDLE_FEATURES_PROP_TIME_0_PACK_STR, buf, 0, self.type, self.length)
        offset = ofproto.OFP_BUNDLE_FEATURES_PROP_TIME_0_SIZE
        for f in [self.sched_accuracy, self.sched_max_future, self.sched_max_past, self.timestamp]:
            f.serialize(buf, offset)
            offset += ofproto.OFP_TIME_SIZE
        return buf