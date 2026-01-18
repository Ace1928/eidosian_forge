import inspect
import itertools
import logging
import sys
import os
import gc
from os_ken import cfg
from os_ken import utils
from os_ken.controller.handler import register_instance, get_dependent_services
from os_ken.controller.controller import Datapath
from os_ken.controller import event
from os_ken.controller.event import EventRequestBase, EventReplyBase
from os_ken.lib import hub
from os_ken.ofproto import ofproto_protocol
def _event_loop(self):
    while self.is_active or not self.events.empty():
        ev, state = self.events.get()
        self._events_sem.release()
        if ev == self._event_stop:
            continue
        handlers = self.get_handlers(ev, state)
        for handler in handlers:
            try:
                handler(ev)
            except hub.TaskExit:
                raise
            except:
                LOG.exception('%s: Exception occurred during handler processing. Backtrace from offending handler [%s] servicing event [%s] follows.', self.name, handler.__name__, ev.__class__.__name__)