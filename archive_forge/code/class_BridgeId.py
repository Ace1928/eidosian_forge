import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
class BridgeId(object):

    def __init__(self, priority, system_id_extension, mac_addr):
        super(BridgeId, self).__init__()
        self.priority = priority
        self.system_id_extension = system_id_extension
        self.mac_addr = mac_addr
        self.value = bpdu.ConfigurationBPDUs.encode_bridge_id(priority, system_id_extension, mac_addr)