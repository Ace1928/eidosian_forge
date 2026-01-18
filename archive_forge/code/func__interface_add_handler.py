import contextlib
import logging
import os
import socket
import struct
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.base.app_manager import OSKenApp
from os_ken.controller.handler import set_ev_cls
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from os_ken.services.protocols.zebra import db
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.server import event as zserver_event
@set_ev_cls(event.EventZebraInterfaceAdd)
def _interface_add_handler(self, ev):
    self.logger.debug('Client %s requested all interfaces', ev.zclient)
    interfaces = db.interface.ip_address_show_all(SESSION)
    self.logger.debug('Server will response interfaces: %s', interfaces)
    for intf in interfaces:
        msg = zebra.ZebraMessage(body=zebra.ZebraInterfaceAdd(ifname=intf.ifname, ifindex=intf.ifindex, status=intf.status, if_flags=intf.flags, ptm_enable=zebra.ZEBRA_IF_PTM_ENABLE_OFF, ptm_status=zebra.ZEBRA_PTM_STATUS_UNKNOWN, metric=intf.metric, ifmtu=intf.ifmtu, ifmtu6=intf.ifmtu6, bandwidth=intf.bandwidth, ll_type=intf.ll_type, hw_addr=intf.hw_addr))
        ev.zclient.send_msg(msg)
        routes = db.route.ip_route_show_all(SESSION, ifindex=intf.ifindex, is_selected=True)
        self.logger.debug('Server will response routes: %s', routes)
        for route in routes:
            dest, _ = route.destination.split('/')
            msg = zebra.ZebraMessage(body=zebra.ZebraInterfaceAddressAdd(ifindex=intf.ifindex, ifc_flags=0, family=None, prefix=route.source, dest=dest))
            ev.zclient.send_msg(msg)