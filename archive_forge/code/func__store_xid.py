import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
def _store_xid(xid, barrier_xid):
    assert xid not in si.results
    assert xid not in si.xids
    assert barrier_xid not in si.barriers
    si.results[xid] = []
    si.xids[xid] = req
    si.barriers[barrier_xid] = xid