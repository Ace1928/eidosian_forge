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
def _add_lo_interface(self):
    intf = db.interface.ip_link_add(SESSION, 'lo')
    if intf:
        self.logger.debug('Added interface "%s": %s', intf.ifname, intf)
    route = db.route.ip_route_add(SESSION, destination='127.0.0.0/8', device='lo', source='127.0.0.1/8', route_type=zebra.ZEBRA_ROUTE_CONNECT)
    if route:
        self.logger.debug('Added route to "%s": %s', route.destination, route)