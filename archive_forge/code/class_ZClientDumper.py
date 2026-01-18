from os_ken.controller.handler import set_ev_cls
from os_ken.lib.packet import zebra
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.client.zclient import ZClient
from os_ken.services.protocols.zebra.client import event as zclient_event
class ZClientDumper(ZClient):

    @set_ev_cls(zclient_event.EventZServConnected)
    def _zserv_connected_handler(self, ev):
        self.logger.info('Zebra server connected to %s: %s', ev.zserv.sock.getpeername(), ev.zserv.sock)

    @set_ev_cls(event.EventZebraRouterIDUpdate)
    def _router_id_update_handler(self, ev):
        self.logger.info('ZEBRA_ROUTER_ID_UPDATE received: %s', ev.__dict__)

    @set_ev_cls(event.EventZebraInterfaceAdd)
    def _interface_add_handler(self, ev):
        self.logger.info('ZEBRA_INTERFACE_ADD received: %s', ev.__dict__)

    @set_ev_cls(event.EventZebraInterfaceAddressAdd)
    def _interface_address_add_handler(self, ev):
        self.logger.info('ZEBRA_INTERFACE_ADDRESS_ADD received: %s', ev.__dict__)

    @set_ev_cls(zclient_event.EventZServDisconnected)
    def _zserv_disconnected_handler(self, ev):
        self.logger.info('Zebra server disconnected: %s', ev.zserv.sock)