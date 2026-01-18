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
@_set_stats_type(ofproto.OFPMP_EXPERIMENTER, OFPExperimenterMultipart)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPExperimenterStatsRequest(OFPExperimenterStatsRequestBase):
    """
    Experimenter multipart request message

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    experimenter     Experimenter ID
    exp_type         Experimenter defined
    data             Experimenter defined additional data
    ================ ======================================================
    """

    def __init__(self, datapath, flags, experimenter, exp_type, data, type_=None):
        super(OFPExperimenterStatsRequest, self).__init__(datapath, flags, experimenter, exp_type, type_)
        self.data = data

    def _serialize_stats_body(self):
        body = OFPExperimenterMultipart(experimenter=self.experimenter, exp_type=self.exp_type, data=self.data)
        self.buf += body.serialize()