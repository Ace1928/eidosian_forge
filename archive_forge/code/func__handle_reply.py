import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
@set_ev_cls(ofp_event.EventOFPErrorMsg, MAIN_DISPATCHER)
def _handle_reply(self, ev):
    msg = ev.msg
    datapath = msg.datapath
    try:
        si = self._switches[datapath.id]
    except KeyError:
        self.logger.error('unknown dpid %s', datapath.id)
        return
    try:
        req = si.xids[msg.xid]
    except KeyError:
        self.logger.error('unknown error xid %s', msg.xid)
        return
    if not isinstance(ev, ofp_event.EventOFPErrorMsg) and (req.reply_cls is None or not isinstance(ev.msg, req.reply_cls)):
        self.logger.error('unexpected reply %s for xid %s', ev, msg.xid)
        return
    try:
        si.results[msg.xid].append(ev.msg)
    except KeyError:
        self.logger.error('unknown error xid %s', msg.xid)