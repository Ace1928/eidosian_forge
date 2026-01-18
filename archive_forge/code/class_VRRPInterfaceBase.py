from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class VRRPInterfaceBase(object):
    """
    interface on which VRRP router works
    vlan_id = None means no vlan.
    NOTE: multiple virtual router can be configured on single port
          See RFC 5798 4.2 Sample Configuration 2
    """

    def __init__(self, mac_address, primary_ip_address, vlan_id=None):
        super(VRRPInterfaceBase, self).__init__()
        self.mac_address = mac_address
        self.primary_ip_address = primary_ip_address
        self.vlan_id = vlan_id

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.mac_address == other.mac_address and (self.primary_ip_address == other.primary_ip_address) and (self.vlan_id == other.vlan_id)

    def __hash__(self):
        return hash((addrconv.mac.text_to_bin(self.mac_address), vrrp.ip_text_to_bin(self.primary_ip_address), self.vlan_id))