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
def _lookup_instance(self, vrid):
    for instance in vrrp_api.vrrp_list(self).instance_list:
        if vrid == instance.config.vrid:
            return instance.instance_name
    return None