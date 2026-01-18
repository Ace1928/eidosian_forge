from os_ken.base.app_manager import OSKenApp
from os_ken.controller.handler import set_ev_cls
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.server.zserver import ZServer
from os_ken.services.protocols.zebra.server import event as zserver_event
class ZServerDumper(OSKenApp):
    _CONTEXTS = {'zserver': ZServer}

    def __init__(self, *args, **kwargs):
        super(ZServerDumper, self).__init__(*args, **kwargs)
        self.zserver = kwargs['zserver']

    @set_ev_cls(zserver_event.EventZClientConnected)
    def _zclient_connected_handler(self, ev):
        self.logger.info('Zebra client connected: %s', ev.zclient.addr)

    @set_ev_cls(zserver_event.EventZClientDisconnected)
    def _zclient_disconnected_handler(self, ev):
        self.logger.info('Zebra client disconnected: %s', ev.zclient.addr)

    @set_ev_cls([event.EventZebraIPv4RouteAdd, event.EventZebraIPv6RouteAdd])
    def _ip_route_add_handler(self, ev):
        self.logger.info('Client %s advertised IP route: %s', ev.zclient.addr, ev.body)

    @set_ev_cls([event.EventZebraIPv4RouteDelete, event.EventZebraIPv6RouteDelete])
    def _ip_route_delete_handler(self, ev):
        self.logger.info('Client %s withdrew IP route: %s', ev.zclient.addr, ev.body)