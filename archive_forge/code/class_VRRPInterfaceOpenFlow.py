from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class VRRPInterfaceOpenFlow(VRRPInterfaceBase):

    def __init__(self, mac_address, primary_ip_address, vlan_id, dpid, port_no):
        super(VRRPInterfaceOpenFlow, self).__init__(mac_address, primary_ip_address, vlan_id)
        self.dpid = dpid
        self.port_no = port_no

    def __str__(self):
        return '%s<%s, %s, %s, %s, %d>' % (self.__class__.__name__, self.mac_address, self.primary_ip_address, self.vlan_id, dpid_lib.dpid_to_str(self.dpid), self.port_no)

    def __eq__(self, other):
        return super(VRRPInterfaceOpenFlow, self).__eq__(other) and self.dpid == other.dpid and (self.port_no == other.port_no)

    def __hash__(self):
        return hash((addrconv.mac.text_to_bin(self.mac_address), vrrp.ip_text_to_bin(self.primary_ip_address), self.vlan_id, self.dpid, self.port_no))