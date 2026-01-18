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
def _submit_event(self, ev):
    self.send_event_to_observers(ev)
    try:
        ev_cls_name = 'Event' + ev.table + ev.event_type
        proxy_ev_cls = getattr(event, ev_cls_name, None)
        if proxy_ev_cls:
            self.send_event_to_observers(proxy_ev_cls(ev))
    except Exception:
        self.logger.exception('Error submitting specific event for OVSDB %s', self.system_id)