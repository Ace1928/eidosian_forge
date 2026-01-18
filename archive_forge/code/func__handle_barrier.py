import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
@set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
def _handle_barrier(self, ev):
    msg = ev.msg
    datapath = msg.datapath
    parser = datapath.ofproto_parser
    try:
        si = self._switches[datapath.id]
    except KeyError:
        self.logger.error('unknown dpid %s', datapath.id)
        return
    try:
        xid = si.barriers.pop(msg.xid)
    except KeyError:
        self.logger.error('unknown barrier xid %s', msg.xid)
        return
    result = si.results.pop(xid)
    req = si.xids.pop(xid)
    is_barrier = isinstance(req.msg, parser.OFPBarrierRequest)
    if req.reply_cls is not None and (not is_barrier):
        self._unobserve_msg(req.reply_cls)
    if is_barrier and req.reply_cls == parser.OFPBarrierReply:
        rep = event.Reply(result=ev.msg)
    elif any((self._is_error(r) for r in result)):
        rep = event.Reply(exception=exception.OFError(result=result))
    elif req.reply_multi:
        rep = event.Reply(result=result)
    elif len(result) == 0:
        rep = event.Reply()
    elif len(result) == 1:
        rep = event.Reply(result=result[0])
    else:
        rep = event.Reply(exception=exception.UnexpectedMultiReply(result=result))
    self.reply_to_request(req, rep)