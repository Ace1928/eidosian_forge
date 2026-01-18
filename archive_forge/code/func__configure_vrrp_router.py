from veth2 by packet generator like packeth
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import mac as lib_mac
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor_linux
from . import vrrp_common
def _configure_vrrp_router(self, vrrp_version, priority, primary_ip_address, ifname, vrid):
    interface = vrrp_event.VRRPInterfaceNetworkDevice(lib_mac.DONTCARE_STR, primary_ip_address, None, ifname)
    self.logger.debug('%s', interface)
    vip = '10.0.%d.1' % vrid
    ip_addresses = [vip]
    config = vrrp_event.VRRPConfig(version=vrrp_version, vrid=vrid, priority=priority, ip_addresses=ip_addresses)
    self.logger.debug('%s', config)
    rep = vrrp_api.vrrp_config(self, interface, config)
    self.logger.debug('%s', rep)
    return rep