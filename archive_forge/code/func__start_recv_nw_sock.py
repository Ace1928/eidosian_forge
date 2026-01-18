import os
import logging
from os_ken.lib import hub, alert
from os_ken.base import app_manager
from os_ken.controller import event
def _start_recv_nw_sock(self, port):
    self.nwsock = hub.socket.socket(hub.socket.AF_INET, hub.socket.SOCK_STREAM)
    self.nwsock.setsockopt(hub.socket.SOL_SOCKET, hub.socket.SO_REUSEADDR, 1)
    self.nwsock.bind(('0.0.0.0', port))
    self.nwsock.listen(5)
    hub.spawn(self._accept_loop_nw_sock)