import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
def _unobserve_msg(self, msg_cls):
    assert msg_cls is not None
    ev_cls = ofp_event.ofp_msg_to_ev_cls(msg_cls)
    assert self._observing_events[ev_cls] > 0
    self._observing_events[ev_cls] -= 1
    if self._observing_events[ev_cls] == 0:
        self.unregister_handler(ev_cls, self._handle_reply)
        self.unobserve_event(ev_cls)
        self.logger.debug('ofctl: stop observing %s', ev_cls)