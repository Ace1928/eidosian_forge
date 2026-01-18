import os
import socket
import struct
from os_ken import cfg
from os_ken.base.app_manager import OSKenApp
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from os_ken.lib.packet import safi as packet_safi
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.client import event as zclient_event
def _service_loop(self):
    while self.is_active:
        self.zserv = ZServer(self)
        self.zserv.start()
        hub.sleep(CONF.retry_interval)
    self.close()