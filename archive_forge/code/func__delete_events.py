import os
import sys
from eventlet import patcher, support
from eventlet.hubs import hub
def _delete_events(self, events):
    del_events = [select.kevent(e.ident, e.filter, select.KQ_EV_DELETE) for e in events]
    self._control(del_events, 0, 0)