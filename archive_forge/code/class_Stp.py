import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
class Stp(app_manager.OSKenApp):
    """ STP(spanning tree) library. """
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION, ofproto_v1_2.OFP_VERSION, ofproto_v1_3.OFP_VERSION]

    def __init__(self):
        super(Stp, self).__init__()
        self.name = 'stplib'
        self._set_logger()
        self.config = {}
        self.bridge_list = {}

    def close(self):
        for dpid in self.bridge_list:
            self._unregister_bridge(dpid)

    def _set_logger(self):
        self.logger.propagate = False
        hdlr = logging.StreamHandler()
        fmt_str = '[STP][%(levelname)s] dpid=%(dpid)s: %(message)s'
        hdlr.setFormatter(logging.Formatter(fmt_str))
        self.logger.addHandler(hdlr)

    def set_config(self, config):
        """ Use this API if you want to set up configuration
             of each bridge and ports.
            Set configuration with 'config' parameter as follows.

             config = {<dpid>: {'bridge': {'priority': <value>,
                                           'sys_ext_id': <value>,
                                           'max_age': <value>,
                                           'hello_time': <value>,
                                           'fwd_delay': <value>}
                                'ports': {<port_no>: {'priority': <value>,
                                                      'path_cost': <value>,
                                                      'enable': <True/False>},
                                          <port_no>: {...},,,}}
                       <dpid>: {...},
                       <dpid>: {...},,,}

             NOTE: You may omit each field.
                    If omitted, a default value is set up.
                   It becomes effective when a bridge starts.

             Default values:
             ------------------------------------------------------
             | bridge | priority   | bpdu.DEFAULT_BRIDGE_PRIORITY |
             |        | sys_ext_id | 0                            |
             |        | max_age    | bpdu.DEFAULT_MAX_AGE         |
             |        | hello_time | bpdu.DEFAULT_HELLO_TIME      |
             |        | fwd_delay  | bpdu.DEFAULT_FORWARD_DELAY   |
             |--------|------------|------------------------------|
             | port   | priority   | bpdu.DEFAULT_PORT_PRIORITY   |
             |        | path_cost  | (Set up automatically        |
             |        |            |   according to link speed.)  |
             |        | enable     | True                         |
             ------------------------------------------------------
        """
        assert isinstance(config, dict)
        self.config = config

    @set_ev_cls(ofp_event.EventOFPStateChange, [handler.MAIN_DISPATCHER, handler.DEAD_DISPATCHER])
    def dispacher_change(self, ev):
        assert ev.datapath is not None
        if ev.state == handler.MAIN_DISPATCHER:
            self._register_bridge(ev.datapath)
        elif ev.state == handler.DEAD_DISPATCHER:
            self._unregister_bridge(ev.datapath.id)

    def _register_bridge(self, dp):
        self._unregister_bridge(dp.id)
        dpid_str = {'dpid': dpid_to_str(dp.id)}
        self.logger.info('Join as stp bridge.', extra=dpid_str)
        try:
            bridge = Bridge(dp, self.logger, self.config.get(dp.id, {}), self.send_event_to_observers)
        except OFPUnknownVersion as message:
            self.logger.error(str(message), extra=dpid_str)
            return
        self.bridge_list[dp.id] = bridge

    def _unregister_bridge(self, dp_id):
        if dp_id in self.bridge_list:
            self.bridge_list[dp_id].delete()
            del self.bridge_list[dp_id]
            self.logger.info('Leave stp bridge.', extra={'dpid': dpid_to_str(dp_id)})

    @set_ev_cls(ofp_event.EventOFPPacketIn, handler.MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        if ev.msg.datapath.id in self.bridge_list:
            bridge = self.bridge_list[ev.msg.datapath.id]
            bridge.packet_in_handler(ev.msg)

    @set_ev_cls(ofp_event.EventOFPPortStatus, handler.MAIN_DISPATCHER)
    def port_status_handler(self, ev):
        dp = ev.msg.datapath
        dpid_str = {'dpid': dpid_to_str(dp.id)}
        port = ev.msg.desc
        reason = ev.msg.reason
        link_down_flg = port.state & 1
        if dp.id in self.bridge_list:
            bridge = self.bridge_list[dp.id]
            if reason is dp.ofproto.OFPPR_ADD:
                self.logger.info('[port=%d] Port add.', port.port_no, extra=dpid_str)
                bridge.port_add(port)
            elif reason is dp.ofproto.OFPPR_DELETE:
                self.logger.info('[port=%d] Port delete.', port.port_no, extra=dpid_str)
                bridge.port_delete(port)
            else:
                assert reason is dp.ofproto.OFPPR_MODIFY
                if bridge.ports_state[port.port_no] == port.state:
                    self.logger.debug('[port=%d] Link status not changed.', port.port_no, extra=dpid_str)
                    return
                if link_down_flg:
                    self.logger.info('[port=%d] Link down.', port.port_no, extra=dpid_str)
                    bridge.link_down(port)
                else:
                    self.logger.info('[port=%d] Link up.', port.port_no, extra=dpid_str)
                    bridge.link_up(port)

    @staticmethod
    def compare_root_path(path_cost1, path_cost2, bridge_id1, bridge_id2, port_id1, port_id2):
        """ Decide the port of the side near a root bridge.
            It is compared by the following priorities.
             1. root path cost
             2. designated bridge ID value
             3. designated port ID value """
        result = Stp._cmp_value(path_cost1, path_cost2)
        if not result:
            result = Stp._cmp_value(bridge_id1, bridge_id2)
            if not result:
                result = Stp._cmp_value(port_id1, port_id2)
        return result

    @staticmethod
    def compare_bpdu_info(my_priority, my_times, rcv_priority, rcv_times):
        """ Check received BPDU is superior to currently held BPDU
             by the following comparison.
             - root bridge ID value
             - root path cost
             - designated bridge ID value
             - designated port ID value
             - times """
        if my_priority is None:
            result = SUPERIOR
        else:
            result = Stp._cmp_value(rcv_priority.root_id.value, my_priority.root_id.value)
            if not result:
                result = Stp.compare_root_path(rcv_priority.root_path_cost, my_priority.root_path_cost, rcv_priority.designated_bridge_id.value, my_priority.designated_bridge_id.value, rcv_priority.designated_port_id.value, my_priority.designated_port_id.value)
                if not result:
                    result1 = Stp._cmp_value(rcv_priority.designated_bridge_id.value, mac.haddr_to_int(my_priority.designated_bridge_id.mac_addr))
                    result2 = Stp._cmp_value(rcv_priority.designated_port_id.value, my_priority.designated_port_id.port_no)
                    if not result1 and (not result2):
                        result = SUPERIOR
                    else:
                        result = Stp._cmp_obj(rcv_times, my_times)
        return result

    @staticmethod
    def _cmp_value(value1, value2):
        result = cmp(value1, value2)
        if result < 0:
            return SUPERIOR
        elif result == 0:
            return REPEATED
        else:
            return INFERIOR

    @staticmethod
    def _cmp_obj(obj1, obj2):
        for key in obj1.__dict__.keys():
            if not hasattr(obj2, key) or getattr(obj1, key) != getattr(obj2, key):
                return SUPERIOR
        return REPEATED