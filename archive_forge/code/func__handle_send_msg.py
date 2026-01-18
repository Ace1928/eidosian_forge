import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
@set_ev_cls(event.SendMsgRequest, MAIN_DISPATCHER)
def _handle_send_msg(self, req):
    msg = req.msg
    datapath = msg.datapath
    parser = datapath.ofproto_parser
    is_barrier = isinstance(msg, parser.OFPBarrierRequest)
    try:
        si = self._switches[datapath.id]
    except KeyError:
        self.logger.error('unknown dpid %s' % (datapath.id,))
        rep = event.Reply(exception=exception.InvalidDatapath(result=datapath.id))
        self.reply_to_request(req, rep)
        return

    def _store_xid(xid, barrier_xid):
        assert xid not in si.results
        assert xid not in si.xids
        assert barrier_xid not in si.barriers
        si.results[xid] = []
        si.xids[xid] = req
        si.barriers[barrier_xid] = xid
    if is_barrier:
        barrier = msg
        datapath.set_xid(barrier)
        _store_xid(barrier.xid, barrier.xid)
    else:
        if req.reply_cls is not None:
            self._observe_msg(req.reply_cls)
        datapath.set_xid(msg)
        barrier = datapath.ofproto_parser.OFPBarrierRequest(datapath)
        datapath.set_xid(barrier)
        _store_xid(msg.xid, barrier.xid)
        if not datapath.send_msg(msg):
            return self._cancel(si, barrier.xid, exception.InvalidDatapath(result=datapath.id))
    if not datapath.send_msg(barrier):
        return self._cancel(si, barrier.xid, exception.InvalidDatapath(result=datapath.id))