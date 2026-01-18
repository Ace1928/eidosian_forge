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
@set_ev_cls([event.EventZebraIPv4RouteAdd, event.EventZebraIPv6RouteAdd])
def _ip_route_add_handler(self, ev):
    self.logger.debug('Client %s advertised IP route: %s', ev.zclient, ev.body)
    for nexthop in ev.body.nexthops:
        route = db.route.ip_route_add(SESSION, destination=ev.body.prefix, gateway=nexthop.addr, ifindex=nexthop.ifindex or 0, route_type=ev.body.route_type)
        if route:
            self.logger.debug('Added route to "%s": %s', route.destination, route)