from os_ken.controller.handler import set_ev_cls
from os_ken.lib.packet import zebra
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.client.zclient import ZClient
from os_ken.services.protocols.zebra.client import event as zclient_event
@set_ev_cls(event.EventZebraRouterIDUpdate)
def _router_id_update_handler(self, ev):
    self.logger.info('ZEBRA_ROUTER_ID_UPDATE received: %s', ev.__dict__)