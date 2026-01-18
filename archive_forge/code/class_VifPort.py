import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
class VifPort(object):

    def __init__(self, port_name, ofport, vif_id, vif_mac, switch):
        super(VifPort, self).__init__()
        self.port_name = port_name
        self.ofport = ofport
        self.vif_id = vif_id
        self.vif_mac = vif_mac
        self.switch = switch

    def __str__(self):
        return 'iface-id=%s, vif_mac=%s, port_name=%s, ofport=%d, bridge_name=%s' % (self.vif_id, self.vif_mac, self.port_name, self.ofport, self.switch.br_name)