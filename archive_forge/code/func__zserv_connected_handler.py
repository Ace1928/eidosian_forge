from os_ken.controller.handler import set_ev_cls
from os_ken.lib.packet import zebra
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.client.zclient import ZClient
from os_ken.services.protocols.zebra.client import event as zclient_event
@set_ev_cls(zclient_event.EventZServConnected)
def _zserv_connected_handler(self, ev):
    self.logger.info('Zebra server connected to %s: %s', ev.zserv.sock.getpeername(), ev.zserv.sock)