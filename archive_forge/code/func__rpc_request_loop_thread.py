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
def _rpc_request_loop_thread(self):
    while True:
        peer, data = self._rpc_events.get()
        msgid, target_method, params = data
        error = None
        result = None
        try:
            if target_method == b'vrrp_config':
                result = self._config(msgid, params)
            elif target_method == b'vrrp_list':
                result = self._list(msgid, params)
            elif target_method == b'vrrp_config_change':
                result = self._config_change(msgid, params)
            else:
                error = 'Unknown method %s' % target_method
        except RPCError as e:
            error = str(e)
        peer._endpoint.send_response(msgid, error=error, result=result)