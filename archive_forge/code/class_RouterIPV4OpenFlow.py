import contextlib
import greenlet
import socket
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.lib import hub
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import arp
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_2
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
class RouterIPV4OpenFlow(RouterIPV4):
    OFP_VERSIONS = [ofproto_v1_2.OFP_VERSION]
    _DROP_TABLE = 0
    _DROP_PRIORITY = 32768 / 2
    _ARP_TABLE = 0
    _ARP_PRIORITY = _DROP_PRIORITY // 2
    _ROUTING_TABLE = 0
    _ROUTING_PRIORITY = _ARP_PRIORITY // 2

    def __init__(self, *args, **kwargs):
        super(RouterIPV4OpenFlow, self).__init__(*args, **kwargs)
        assert isinstance(self.interface, vrrp_event.VRRPInterfaceOpenFlow)

    def _get_dp(self):
        return utils.get_dp(self, self.interface.dpid)

    def start(self):
        dp = self._get_dp()
        assert dp
        self._uninstall_route_rule(dp)
        self._uninstall_arp_rule(dp)
        self._uninstall_drop_rule(dp)
        self._install_drop_rule(dp)
        self._install_arp_rule(dp)
        self._install_route_rule(dp)
        super(RouterIPV4OpenFlow, self).start()

    def _initialized_to_master(self):
        self.logger.debug('initialized to master')
        self._master()

    def _become_master(self):
        self.logger.debug('become master')
        self._master()

    def _master(self):
        dp = self._get_dp()
        if dp is None:
            return
        self._uninstall_drop_rule(dp)
        self._send_garp(dp)

    def _become_backup(self):
        self.logger.debug('become backup')
        dp = self._get_dp()
        if dp is None:
            return
        self._install_drop_rule(dp)

    def _shutdowned(self):
        dp = self._get_dp()
        if dp is None:
            return
        self._uninstall_route_rule(dp)
        self._uninstall_arp_rule(dp)
        self._uninstall_drop_rule(dp)

    @handler.set_ev_cls(ofp_event.EventOFPPacketIn, handler.MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        dpid = datapath.dpid
        if dpid != self.interface.dpid:
            return
        for field in msg.match.fields:
            header = field.header
            if header == ofproto.OXM_OF_IN_PORT:
                if field.value != self.interface.port_no:
                    return
                break
        self._arp_process(msg.data)

    def _drop_match(self, dp):
        kwargs = {}
        kwargs['in_port'] = self.interface.port_no
        kwargs['eth_dst'] = vrrp.vrrp_ipv4_src_mac_address(self.config.vrid)
        if self.interface.vlan_id is not None:
            kwargs['vlan_vid'] = self.interface.vlan_id
        return dp.ofproto_parser.OFPMatch(**kwargs)

    def _install_drop_rule(self, dp):
        match = self._drop_match(dp)
        utils.dp_flow_mod(dp, self._DROP_TABLE, dp.ofproto.OFPFC_ADD, self._DROP_PRIORITY, match, [])

    def _uninstall_drop_rule(self, dp):
        match = self._drop_match(dp)
        utils.dp_flow_mod(dp, self._DROP_TABLE, dp.ofproto.OFPFC_DELETE_STRICT, self._DROP_PRIORITY, match, [])

    def _arp_match(self, dp):
        kwargs = {}
        kwargs['in_port'] = self.interface.port_no
        kwargs['eth_dst'] = mac_lib.BROADCAST_STR
        kwargs['eth_type'] = ether.ETH_TYPE_ARP
        if self.interface.vlan_id is not None:
            kwargs['vlan_vid'] = self.interface.vlan_id
        kwargs['arp_op'] = arp.ARP_REQUEST
        kwargs['arp_tpa'] = vrrp.vrrp_ipv4_src_mac_address(self.config.vrid)
        return dp.ofproto_parser.OFPMatch(**kwargs)

    def _install_arp_rule(self, dp):
        ofproto = dp.ofproto
        ofproto_parser = dp.ofproto_parser
        match = self._arp_match(dp)
        actions = [ofproto_parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        instructions = [ofproto_parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        utils.dp_flow_mod(dp, self._ARP_TABLE, dp.fproto.OFPFC_ADD, self._ARP_PRIORITY, match, instructions)

    def _uninstall_arp_rule(self, dp):
        match = self._arp_match(dp)
        utils.dp_flow_mod(dp, self._ARP_TABLE, dp.fproto.OFPFC_DELETE_STRICT, self._ARP_PRIORITY, match, [])

    def _install_route_rule(self, dp):
        self.logger.debug('TODO:_install_router_rule')

    def _uninstall_route_rule(self, dp):
        self.logger.debug('TODO:_uninstall_router_rule')