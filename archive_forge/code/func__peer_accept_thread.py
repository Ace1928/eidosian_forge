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
def _peer_accept_thread(self):
    server = hub.StreamServer(('', self.CONF.vrrp_rpc_port), self.peer_accept_handler)
    server.serve_forever()