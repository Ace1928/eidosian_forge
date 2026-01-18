from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
def _send_vrrp_packet_received(self, packet_data):
    valid = self._parse_received_packet(packet_data)
    if valid is True:
        self.statistics.rx_vrrp_packets += 1
    else:
        self.statistics.rx_vrrp_invalid_packets += 1