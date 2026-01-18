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
@OFPMeterBandHeader.register_meter_band_type(ofproto.OFPMBT_EXPERIMENTER, ofproto.OFP_METER_BAND_EXPERIMENTER_SIZE)
class OFPMeterBandExperimenter(OFPMeterBandHeader):

    def __init__(self, rate=0, burst_size=0, experimenter=None, type_=None, len_=None):
        super(OFPMeterBandExperimenter, self).__init__()
        self.rate = rate
        self.burst_size = burst_size
        self.experimenter = experimenter

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_METER_BAND_EXPERIMENTER_PACK_STR, buf, offset, self.type, self.len, self.rate, self.burst_size, self.experimenter)

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, rate, burst_size, experimenter = struct.unpack_from(ofproto.OFP_METER_BAND_EXPERIMENTER_PACK_STR, buf, offset)
        assert cls.cls_meter_band_type == type_
        assert cls.cls_meter_band_len == len_
        return cls(rate, burst_size, experimenter)