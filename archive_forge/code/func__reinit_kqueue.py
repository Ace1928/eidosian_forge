import os
import sys
from eventlet import patcher, support
from eventlet.hubs import hub
def _reinit_kqueue(self):
    self.kqueue.close()
    self._init_kqueue()
    events = [e for i in self._events.values() for e in i.values()]
    self.kqueue.control(events, 0, 0)