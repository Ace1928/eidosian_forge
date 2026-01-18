import logging
import warnings
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
import os_ken.exception as os_ken_exc
from os_ken.lib.dpid import dpid_to_str
class DPSet(app_manager.OSKenApp):
    """
    DPSet application manages a set of switches (datapaths)
    connected to this controller.

    Usage Example::

        # ...(snip)...
        from os_ken.controller import dpset


        class MyApp(app_manager.OSKenApp):
            _CONTEXTS = {
                'dpset': dpset.DPSet,
            }

            def __init__(self, *args, **kwargs):
                super(MyApp, self).__init__(*args, **kwargs)
                # Stores DPSet instance to call its API in this app
                self.dpset = kwargs['dpset']

            def _my_handler(self):
                # Get the datapath object which has the given dpid
                dpid = 1
                dp = self.dpset.get(dpid)
                if dp is None:
                    self.logger.info('No such datapath: dpid=%d', dpid)
    """

    def __init__(self, *args, **kwargs):
        super(DPSet, self).__init__(*args, **kwargs)
        self.name = 'dpset'
        self.dps = {}
        self.port_state = {}

    def _register(self, dp):
        LOG.debug('DPSET: register datapath %s', dp)
        assert dp.id is not None
        send_dp_reconnected = False
        if dp.id in self.dps:
            self.logger.warning('DPSET: Multiple connections from %s', dpid_to_str(dp.id))
            self.logger.debug('DPSET: Forgetting datapath %s', self.dps[dp.id])
            self.dps[dp.id].close()
            self.logger.debug('DPSET: New datapath %s', dp)
            send_dp_reconnected = True
        self.dps[dp.id] = dp
        if dp.id not in self.port_state:
            self.port_state[dp.id] = PortState()
            ev = EventDP(dp, True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for port in dp.ports.values():
                    self._port_added(dp, port)
                    ev.ports.append(port)
            self.send_event_to_observers(ev)
        if send_dp_reconnected:
            ev = EventDPReconnected(dp)
            ev.ports = self.port_state.get(dp.id, {}).values()
            self.send_event_to_observers(ev)

    def _unregister(self, dp):
        if dp not in self.dps.values():
            return
        LOG.debug('DPSET: unregister datapath %s', dp)
        assert self.dps[dp.id] == dp
        ev = EventDP(dp, False)
        for port in list(self.port_state.get(dp.id, {}).values()):
            self._port_deleted(dp, port)
            ev.ports.append(port)
        self.send_event_to_observers(ev)
        del self.dps[dp.id]
        del self.port_state[dp.id]

    def get(self, dp_id):
        """
        This method returns the os_ken.controller.controller.Datapath
        instance for the given Datapath ID.
        """
        return self.dps.get(dp_id)

    def get_all(self):
        """
        This method returns a list of tuples which represents
        instances for switches connected to this controller.
        The tuple consists of a Datapath ID and an instance of
        os_ken.controller.controller.Datapath.

        A return value looks like the following::

            [ (dpid_A, Datapath_A), (dpid_B, Datapath_B), ... ]
        """
        return list(self.dps.items())

    def _port_added(self, datapath, port):
        self.port_state[datapath.id].add(port.port_no, port)

    def _port_deleted(self, datapath, port):
        self.port_state[datapath.id].remove(port.port_no)

    @set_ev_cls(ofp_event.EventOFPStateChange, [handler.MAIN_DISPATCHER, handler.DEAD_DISPATCHER])
    def dispatcher_change(self, ev):
        datapath = ev.datapath
        assert datapath is not None
        if ev.state == handler.MAIN_DISPATCHER:
            self._register(datapath)
        elif ev.state == handler.DEAD_DISPATCHER:
            self._unregister(datapath)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, handler.CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        if datapath.ofproto.OFP_VERSION < 4:
            datapath.ports = msg.ports

    @set_ev_cls(ofp_event.EventOFPPortStatus, handler.MAIN_DISPATCHER)
    def port_status_handler(self, ev):
        msg = ev.msg
        reason = msg.reason
        datapath = msg.datapath
        port = msg.desc
        ofproto = datapath.ofproto
        if reason == ofproto.OFPPR_ADD:
            LOG.debug('DPSET: A port was added.' + '(datapath id = %s, port number = %s)', dpid_to_str(datapath.id), port.port_no)
            self._port_added(datapath, port)
            self.send_event_to_observers(EventPortAdd(datapath, port))
        elif reason == ofproto.OFPPR_DELETE:
            LOG.debug('DPSET: A port was deleted.' + '(datapath id = %s, port number = %s)', dpid_to_str(datapath.id), port.port_no)
            self._port_deleted(datapath, port)
            self.send_event_to_observers(EventPortDelete(datapath, port))
        else:
            assert reason == ofproto.OFPPR_MODIFY
            LOG.debug('DPSET: A port was modified.' + '(datapath id = %s, port number = %s)', dpid_to_str(datapath.id), port.port_no)
            self.port_state[datapath.id].modify(port.port_no, port)
            self.send_event_to_observers(EventPortModify(datapath, port))

    def get_port(self, dpid, port_no):
        """
        This method returns the os_ken.controller.dpset.PortState
        instance for the given Datapath ID and the port number.
        Raises os_ken_exc.PortNotFound if no such a datapath connected to
        this controller or no such a port exists.
        """
        try:
            return self.port_state[dpid][port_no]
        except KeyError:
            raise os_ken_exc.PortNotFound(dpid=dpid, port=port_no, network_id=None)

    def get_ports(self, dpid):
        """
        This method returns a list of os_ken.controller.dpset.PortState
        instances for the given Datapath ID.
        Raises KeyError if no such a datapath connected to this controller.
        """
        return list(self.port_state[dpid].values())