from os_ken.base.app_manager import OSKenApp
from os_ken.controller.handler import set_ev_cls
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.server.zserver import ZServer
from os_ken.services.protocols.zebra.server import event as zserver_event
@set_ev_cls(zserver_event.EventZClientConnected)
def _zclient_connected_handler(self, ev):
    self.logger.info('Zebra client connected: %s', ev.zclient.addr)