import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
@set_ev_cls(ofp_event.EventOFPStateChange, DEAD_DISPATCHER)
def _handle_dead(self, ev):
    datapath = ev.datapath
    id = datapath.id
    self.logger.debug('del dpid %s datapath %s', id, datapath)
    if id is None:
        return
    try:
        info = self._switches[id]
    except KeyError:
        return
    if info.datapath is datapath:
        self.logger.debug('forget info %s', info)
        self._switches.pop(id)
        for xid in list(info.barriers):
            self._cancel(info, xid, exception.InvalidDatapath(result=id))