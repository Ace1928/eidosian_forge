import contextlib
import greenlet
import socket
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.lib import hub
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import arp
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_2
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
class RouterBase(app_manager.OSKenApp):

    def _router_name(self, config, interface):
        ip_version = 'ipv6' if config.is_ipv6 else 'ipv4'
        return '%s-%s-%d-%s' % (self.__class__.__name__, str(interface), config.vrid, ip_version)

    def __init__(self, *args, **kwargs):
        super(RouterBase, self).__init__(*args, **kwargs)
        self.instance_name = kwargs['name']
        self.monitor_name = kwargs['monitor_name']
        self.config = kwargs['config']
        self.interface = kwargs['interface']
        self.name = self._router_name(self.config, self.interface)

    def _transmit(self, data):
        vrrp_api.vrrp_transmit(self, self.monitor_name, data)

    def _initialized(self):
        self.logger.debug('initialized')

    def _initialized_to_master(self):
        self.logger.debug('initialized to master')

    def _become_master(self):
        self.logger.debug('become master')

    def _become_backup(self):
        self.logger.debug('become backup')

    def _shutdowned(self):
        self.logger.debug('shutdowned')

    @handler.set_ev_handler(vrrp_event.EventVRRPStateChanged)
    def vrrp_state_changed_handler(self, ev):
        old_state = ev.old_state
        new_state = ev.new_state
        self.logger.debug('sample router %s -> %s', old_state, new_state)
        if new_state == vrrp_event.VRRP_STATE_MASTER:
            if old_state == vrrp_event.VRRP_STATE_INITIALIZE:
                self._initialized_to_master()
            elif old_state == vrrp_event.VRRP_STATE_BACKUP:
                self._become_master()
        elif new_state == vrrp_event.VRRP_STATE_BACKUP:
            self._become_backup()
        elif new_state == vrrp_event.VRRP_STATE_INITIALIZE:
            if old_state is None:
                self._initialized()
            else:
                self._shutdowned()
        else:
            raise ValueError('invalid vrrp state %s' % new_state)