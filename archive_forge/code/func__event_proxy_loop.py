import collections
import errno
import uuid
from ovs import jsonrpc
from ovs import poller
from ovs import reconnect
from ovs import stream
from ovs import timeval
from ovs.db import idl
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.services.protocols.ovsdb import event
from os_ken.services.protocols.ovsdb import model
def _event_proxy_loop(self):
    while self.is_active:
        events = self._idl.events
        if not events:
            hub.sleep(0.1)
            continue
        for e in events:
            ev = e[0]
            args = e[1]
            self._submit_event(ev(self.system_id, *args))
        hub.sleep(0)