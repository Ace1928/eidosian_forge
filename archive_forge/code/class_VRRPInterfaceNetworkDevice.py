from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class VRRPInterfaceNetworkDevice(VRRPInterfaceBase):

    def __init__(self, mac_address, primary_ip_address, vlan_id, device_name):
        super(VRRPInterfaceNetworkDevice, self).__init__(mac_address, primary_ip_address, vlan_id)
        self.device_name = device_name

    def __str__(self):
        return '%s<%s, %s, %s, %s>' % (self.__class__.__name__, self.mac_address, self.primary_ip_address, self.vlan_id, self.device_name)

    def __eq__(self, other):
        return super(VRRPInterfaceNetworkDevice, self).__eq__(other) and self.device_name == other.device_name

    def __hash__(self):
        return hash((addrconv.mac.text_to_bin(self.mac_address), vrrp.ip_text_to_bin(self.primary_ip_address), self.vlan_id, self.device_name))