import logging
import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.topology import event
from os_ken.topology import switches
def _switch_request_async(self, interval):
    while self.is_active:
        request = event.EventSwitchRequest()
        LOG.debug('switch_request async %s thread(%s)', request, id(hub.getcurrent()))
        self.send_event(request.dst, request)
        start = time.time()
        busy = interval / 2
        i = 0
        while i < busy:
            if time.time() > start + i:
                i += 1
                LOG.debug('  thread is busy... %s/%s thread(%s)', i, busy, id(hub.getcurrent()))
        LOG.debug('  thread yield to switch_reply handler. thread(%s)', id(hub.getcurrent()))
        hub.sleep(0)
        LOG.debug('  thread get back. thread(%s)', id(hub.getcurrent()))
        hub.sleep(interval - busy)