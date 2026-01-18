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
@_set_msg_type(ofproto.OFPT_PORT_MOD)
class OFPPortMod(MsgBase):
    """
    Port modification message

    The controller sneds this message to modify the behavior of the port.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    port_no          Port number to modify
    hw_addr          The hardware address that must be the same as hw_addr
                     of ``OFPPort`` of ``OFPSwitchFeatures``
    config           Bitmap of configuration flags.

                     | OFPPC_PORT_DOWN
                     | OFPPC_NO_RECV
                     | OFPPC_NO_FWD
                     | OFPPC_NO_PACKET_IN
    mask             Bitmap of configuration flags above to be changed
    advertise        Bitmap of the following flags.

                     | OFPPF_10MB_HD
                     | OFPPF_10MB_FD
                     | OFPPF_100MB_HD
                     | OFPPF_100MB_FD
                     | OFPPF_1GB_HD
                     | OFPPF_1GB_FD
                     | OFPPF_10GB_FD
                     | OFPPF_40GB_FD
                     | OFPPF_100GB_FD
                     | OFPPF_1TB_FD
                     | OFPPF_OTHER
                     | OFPPF_COPPER
                     | OFPPF_FIBER
                     | OFPPF_AUTONEG
                     | OFPPF_PAUSE
                     | OFPPF_PAUSE_ASYM
    ================ ======================================================

    Example::

        def send_port_mod(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            port_no = 3
            hw_addr = 'fa:c8:e8:76:1d:7e'
            config = 0
            mask = (ofp.OFPPC_PORT_DOWN | ofp.OFPPC_NO_RECV |
                    ofp.OFPPC_NO_FWD | ofp.OFPPC_NO_PACKET_IN)
            advertise = (ofp.OFPPF_10MB_HD | ofp.OFPPF_100MB_FD |
                         ofp.OFPPF_1GB_FD | ofp.OFPPF_COPPER |
                         ofp.OFPPF_AUTONEG | ofp.OFPPF_PAUSE |
                         ofp.OFPPF_PAUSE_ASYM)
            req = ofp_parser.OFPPortMod(datapath, port_no, hw_addr, config,
                                        mask, advertise)
            datapath.send_msg(req)
    """
    _TYPE = {'ascii': ['hw_addr']}

    def __init__(self, datapath, port_no=0, hw_addr='00:00:00:00:00:00', config=0, mask=0, advertise=0):
        super(OFPPortMod, self).__init__(datapath)
        self.port_no = port_no
        self.hw_addr = hw_addr
        self.config = config
        self.mask = mask
        self.advertise = advertise

    def _serialize_body(self):
        msg_pack_into(ofproto.OFP_PORT_MOD_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.port_no, addrconv.mac.text_to_bin(self.hw_addr), self.config, self.mask, self.advertise)