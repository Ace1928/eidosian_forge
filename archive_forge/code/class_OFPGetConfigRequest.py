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
@_set_msg_type(ofproto.OFPT_GET_CONFIG_REQUEST)
class OFPGetConfigRequest(MsgBase):
    """
    Get config request message

    The controller sends a get config request to query configuration
    parameters in the switch.

    Example::

        def send_get_config_request(self, datapath):
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPGetConfigRequest(datapath)
            datapath.send_msg(req)
    """

    def __init__(self, datapath):
        super(OFPGetConfigRequest, self).__init__(datapath)