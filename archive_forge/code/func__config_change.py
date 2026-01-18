from os_ken import cfg
import socket
import netaddr
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.lib import rpc
from os_ken.lib import hub
from os_ken.lib import mac
def _config_change(self, msgid, params):
    self.logger.debug('handle vrrp_config_change request')
    try:
        config_values = params[0]
    except:
        raise RPCError('parameters are missing')
    vrid = config_values.get('vrid')
    instance_name = self._lookup_instance(vrid)
    if not instance_name:
        raise RPCError('vrid %d is not found' % vrid)
    priority = config_values.get('priority')
    interval = config_values.get('advertisement_interval')
    vrrp_api.vrrp_config_change(self, instance_name, priority=priority, advertisement_interval=interval)
    return {}