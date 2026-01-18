import logging
import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.topology import event
from os_ken.topology import switches
def _switch_request_sync(self, interval):
    while self.is_active:
        request = event.EventSwitchRequest()
        LOG.debug('switch_request sync %s thread(%s)', request, id(hub.getcurrent()))
        reply = self.send_request(request)
        LOG.debug('switch_reply sync %s', reply)
        if len(reply.switches) > 0:
            for sw in reply.switches:
                LOG.debug('  %s', sw)
        hub.sleep(interval)