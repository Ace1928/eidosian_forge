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
@set_ev_cls(event.EventZebraHello)
def _hello_handler(self, ev):
    if ev.body is None:
        self.logger.debug('Client %s says hello.', ev.zclient)
        return
    ev.zclient.route_type = ev.body.route_type
    self.logger.debug('Client %s says hello and bids fair to announce only %s routes', ev.zclient, ev.body.route_type)