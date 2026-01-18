from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib.packet import vrrp
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.services.protocols.vrrp import monitor
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
def _ofp_match(self, ofproto_parser):
    is_ipv6 = vrrp.is_ipv6(self.config.ip_addresses[0])
    kwargs = {}
    kwargs['in_port'] = self.interface.port_no
    if is_ipv6:
        kwargs['eth_dst'] = vrrp.VRRP_IPV6_DST_MAC_ADDRESS
        kwargs['eth_src'] = vrrp.vrrp_ipv6_src_mac_address(self.config.vrid)
        kwargs['eth_type'] = ether.ETH_TYPE_IPV6
        kwargs['ipv6_dst'] = vrrp.VRRP_IPV6_DST_ADDRESS
    else:
        kwargs['eth_dst'] = vrrp.VRRP_IPV4_DST_MAC_ADDRESS
        kwargs['eth_src'] = vrrp.vrrp_ipv4_src_mac_address(self.config.vrid)
        kwargs['eth_type'] = ether.ETH_TYPE_IP
        kwargs['ipv4_dst'] = vrrp.VRRP_IPV4_DST_ADDRESS
    if self.interface.vlan_id is not None:
        kwargs['vlan_vid'] = self.interface.vlan_id
    kwargs['ip_proto'] = inet.IPPROTO_VRRP
    return ofproto_parser.OFPMatch(**kwargs)