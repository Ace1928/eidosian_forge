import os
import logging
from os_ken.lib import hub, alert
from os_ken.base import app_manager
from os_ken.controller import event
def _start_recv(self):
    if os.path.exists(SOCKFILE):
        os.unlink(SOCKFILE)
    self.sock = hub.socket.socket(hub.socket.AF_UNIX, hub.socket.SOCK_DGRAM)
    self.sock.bind(SOCKFILE)
    hub.spawn(self._recv_loop)