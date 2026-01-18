import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
class DPIDs(object):
    """dpid -> port_no -> remote_dpid"""

    def __init__(self, f):
        super(DPIDs, self).__init__()
        self.dpids = collections.defaultdict(dict)
        self.send_event = f

    def list_ports(self, dpid):
        return self.dpids[dpid]

    def _add_remote_dpid(self, dpid, port_no, remote_dpid):
        self.dpids[dpid][port_no] = remote_dpid
        self.send_event(EventTunnelPort(dpid, port_no, remote_dpid, True))

    def add_remote_dpid(self, dpid, port_no, remote_dpid):
        if port_no in self.dpids[dpid]:
            raise os_ken_exc.PortAlreadyExist(dpid=dpid, port=port_no, network_id=None)
        self._add_remote_dpid(dpid, port_no, remote_dpid)

    def update_remote_dpid(self, dpid, port_no, remote_dpid):
        remote_dpid_ = self.dpids[dpid].get(port_no)
        if remote_dpid_ is None:
            self._add_remote_dpid(dpid, port_no, remote_dpid)
        elif remote_dpid_ != remote_dpid:
            raise os_ken_exc.RemoteDPIDAlreadyExist(dpid=dpid, port=port_no, remote_dpid=remote_dpid)

    def get_remote_dpid(self, dpid, port_no):
        try:
            return self.dpids[dpid][port_no]
        except KeyError:
            raise os_ken_exc.PortNotFound(dpid=dpid, port=port_no)

    def delete_port(self, dpid, port_no):
        try:
            remote_dpid = self.dpids[dpid][port_no]
            self.send_event(EventTunnelPort(dpid, port_no, remote_dpid, False))
            del self.dpids[dpid][port_no]
        except KeyError:
            raise os_ken_exc.PortNotFound(dpid=dpid, port=port_no)

    def get_port(self, dpid, remote_dpid):
        try:
            dp = self.dpids[dpid]
        except KeyError:
            raise os_ken_exc.PortNotFound(dpid=dpid, port=None, network_id=None)
        res = [port_no for port_no, remote_dpid_ in dp.items() if remote_dpid_ == remote_dpid]
        assert len(res) <= 1
        if len(res) == 0:
            raise os_ken_exc.PortNotFound(dpid=dpid, port=None, network_id=None)
        return res[0]