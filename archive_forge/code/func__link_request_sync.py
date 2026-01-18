import logging
import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.topology import event
from os_ken.topology import switches
def _link_request_sync(self, interval):
    while self.is_active:
        request = event.EventLinkRequest()
        LOG.debug('link_request sync %s thread(%s)', request, id(hub.getcurrent()))
        reply = self.send_request(request)
        LOG.debug('link_reply sync %s', reply)
        if len(reply.links) > 0:
            for link in reply.links:
                LOG.debug('  %s', link)
        hub.sleep(interval)