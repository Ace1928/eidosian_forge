import logging
import struct
import time
from os_ken import cfg
from collections import defaultdict
from os_ken.topology import event
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from os_ken.exception import OSKenException
from os_ken.lib import addrconv, hub
from os_ken.lib.mac import DONTCARE_STR
from os_ken.lib.dpid import dpid_to_str, str_to_dpid
from os_ken.lib.port_no import port_no_to_str
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.packet import lldp, ether_types
from os_ken.ofproto.ether import ETH_TYPE_LLDP
from os_ken.ofproto.ether import ETH_TYPE_CFM
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
class HostState(dict):

    def __init__(self):
        super(HostState, self).__init__()

    def add(self, host):
        mac = host.mac
        self.setdefault(mac, host)

    def update_ip(self, host, ip_v4=None, ip_v6=None):
        mac = host.mac
        host = None
        if mac in self:
            host = self[mac]
        if not host:
            return
        if ip_v4 is not None:
            if ip_v4 in host.ipv4:
                host.ipv4.remove(ip_v4)
            host.ipv4.append(ip_v4)
        if ip_v6 is not None:
            if ip_v6 in host.ipv6:
                host.ipv6.remove(ip_v6)
            host.ipv6.append(ip_v6)

    def get_by_dpid(self, dpid):
        result = []
        for mac in self:
            host = self[mac]
            if host.port.dpid == dpid:
                result.append(host)
        return result