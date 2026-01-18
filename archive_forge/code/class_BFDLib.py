import logging
import time
import random
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.ofproto.ether import ETH_TYPE_IP, ETH_TYPE_ARP
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import inet
from os_ken.lib import hub
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib.packet import arp
from os_ken.lib.packet.arp import ARP_REQUEST, ARP_REPLY
class BFDLib(app_manager.OSKenApp):
    """
    BFD daemon library.

    Add this library as a context in your app and use ``add_bfd_session``
    function to establish a BFD session.

    Example::

        from os_ken.base import app_manager
        from os_ken.controller.handler import set_ev_cls
        from os_ken.ofproto import ofproto_v1_3
        from os_ken.lib import bfdlib
        from os_ken.lib.packet import bfd

        class Foo(app_manager.OSKenApp):
            OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

            _CONTEXTS = {
                'bfdlib': bfdlib.BFDLib
            }

            def __init__(self, *args, **kwargs):
                super(Foo, self).__init__(*args, **kwargs)
                self.bfdlib = kwargs['bfdlib']
                self.my_discr =                     self.bfdlib.add_bfd_session(dpid=1,
                                                ofport=1,
                                                src_mac="00:23:45:67:89:AB",
                                                src_ip="192.168.1.1")

            @set_ev_cls(bfdlib.EventBFDSessionStateChanged)
            def bfd_state_handler(self, ev):
                if ev.session.my_discr != self.my_discr:
                    return

                if ev.new_state == bfd.BFD_STATE_DOWN:
                    print "BFD Session=%d is DOWN!" % ev.session.my_discr
                elif ev.new_state == bfd.BFD_STATE_UP:
                    print "BFD Session=%d is UP!" % ev.session.my_discr
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _EVENTS = [EventBFDSessionStateChanged]

    def __init__(self, *args, **kwargs):
        super(BFDLib, self).__init__(*args, **kwargs)
        self.session = {}

    def close(self):
        pass

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        for s in self.session.values():
            if s.dpid == datapath.id:
                s.datapath = datapath
        match = parser.OFPMatch(eth_type=ETH_TYPE_ARP)
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 65535, match, actions)
        match = parser.OFPMatch(eth_type=ETH_TYPE_IP, ip_proto=inet.IPPROTO_UDP, udp_dst=3784)
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 65535, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        if arp.arp in pkt:
            arp_pkt = ARPPacket.arp_parse(msg.data)
            if arp_pkt.opcode == ARP_REQUEST:
                for s in self.session.values():
                    if s.dpid == datapath.id and s.ofport == in_port and (s.src_ip == arp_pkt.dst_ip):
                        ans = ARPPacket.arp_packet(ARP_REPLY, s.src_mac, s.src_ip, arp_pkt.src_mac, arp_pkt.src_ip)
                        actions = [parser.OFPActionOutput(in_port)]
                        out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=ofproto.OFPP_CONTROLLER, actions=actions, data=ans)
                        datapath.send_msg(out)
                        return
            return
        if ipv4.ipv4 not in pkt or udp.udp not in pkt:
            return
        udp_hdr = pkt.get_protocols(udp.udp)[0]
        if udp_hdr.dst_port != BFD_CONTROL_UDP_PORT:
            return
        self.recv_bfd_pkt(datapath, in_port, msg.data)

    def add_bfd_session(self, dpid, ofport, src_mac, src_ip, dst_mac='FF:FF:FF:FF:FF:FF', dst_ip='255.255.255.255', auth_type=0, auth_keys=None):
        """
        Establish a new BFD session and return My Discriminator of new session.

        Configure the BFD session with the following arguments.

        ================ ======================================================
        Argument         Description
        ================ ======================================================
        dpid             Datapath ID of the BFD interface.
        ofport           Openflow port number of the BFD interface.
        src_mac          Source MAC address of the BFD interface.
        src_ip           Source IPv4 address of the BFD interface.
        dst_mac          (Optional) Destination MAC address of the BFD
                         interface.
        dst_ip           (Optional) Destination IPv4 address of the BFD
                         interface.
        auth_type        (Optional) Authentication type.
        auth_keys        (Optional) A dictionary of authentication key chain
                         which key is an integer of *Auth Key ID* and value
                         is a string of *Password* or *Auth Key*.
        ================ ======================================================

        Example::

            add_bfd_session(dpid=1,
                            ofport=1,
                            src_mac="01:23:45:67:89:AB",
                            src_ip="192.168.1.1",
                            dst_mac="12:34:56:78:9A:BC",
                            dst_ip="192.168.1.2",
                            auth_type=bfd.BFD_AUTH_KEYED_SHA1,
                            auth_keys={1: "secret key 1",
                                       2: "secret key 2"})
        """
        auth_keys = auth_keys if auth_keys else {}
        while True:
            my_discr = random.randint(1, UINT32_MAX)
            src_port = random.randint(49152, 65535)
            if my_discr in self.session:
                continue
            unique_flag = True
            for s in self.session.values():
                if s.your_discr == my_discr or s.src_port == src_port:
                    unique_flag = False
                    break
            if unique_flag:
                break
        sess = BFDSession(app=self, my_discr=my_discr, dpid=dpid, ofport=ofport, src_mac=src_mac, src_ip=src_ip, src_port=src_port, dst_mac=dst_mac, dst_ip=dst_ip, auth_type=auth_type, auth_keys=auth_keys)
        self.session[my_discr] = sess
        return my_discr

    def recv_bfd_pkt(self, datapath, in_port, data):
        pkt = packet.Packet(data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype != ETH_TYPE_IP:
            return
        ip_pkt = pkt.get_protocols(ipv4.ipv4)[0]
        if ip_pkt.ttl != 255:
            return
        bfd_pkt = BFDPacket.bfd_parse(data)
        if not isinstance(bfd_pkt, bfd.bfd):
            return
        if bfd_pkt.ver != 1:
            return
        if bfd_pkt.flags & bfd.BFD_FLAG_AUTH_PRESENT:
            if bfd_pkt.length < 26:
                return
        elif bfd_pkt.length < 24:
            return
        if bfd_pkt.detect_mult == 0:
            return
        if bfd_pkt.flags & bfd.BFD_FLAG_MULTIPOINT:
            return
        if bfd_pkt.my_discr == 0:
            return
        if bfd_pkt.your_discr != 0 and bfd_pkt.your_discr not in self.session:
            return
        if bfd_pkt.your_discr == 0 and bfd_pkt.state not in [bfd.BFD_STATE_ADMIN_DOWN, bfd.BFD_STATE_DOWN]:
            return
        sess_my_discr = None
        if bfd_pkt.your_discr == 0:
            for s in self.session.values():
                if s.dpid == datapath.id and s.ofport == in_port:
                    sess_my_discr = s.my_discr
                    break
            if sess_my_discr is None:
                return
        else:
            sess_my_discr = bfd_pkt.your_discr
        sess = self.session[sess_my_discr]
        if bfd_pkt.flags & bfd.BFD_FLAG_AUTH_PRESENT and sess._auth_type == 0:
            return
        if bfd_pkt.flags & bfd.BFD_FLAG_AUTH_PRESENT == 0 and sess._auth_type != 0:
            return
        if bfd_pkt.flags & bfd.BFD_FLAG_AUTH_PRESENT:
            if sess._auth_type == 0:
                return
            if bfd_pkt.auth_cls.auth_type != sess._auth_type:
                return
            if sess._auth_type in [bfd.BFD_AUTH_KEYED_MD5, bfd.BFD_AUTH_METICULOUS_KEYED_MD5, bfd.BFD_AUTH_KEYED_SHA1, bfd.BFD_AUTH_METICULOUS_KEYED_SHA1]:
                if sess._auth_seq_known:
                    if bfd_pkt.auth_cls.seq < sess._rcv_auth_seq:
                        return
                    if sess._auth_type in [bfd.BFD_AUTH_METICULOUS_KEYED_MD5, bfd.BFD_AUTH_METICULOUS_KEYED_SHA1]:
                        if bfd_pkt.auth_cls.seq <= sess._rcv_auth_seq:
                            return
                    if bfd_pkt.auth_cls.seq > sess._rcv_auth_seq + 3 * sess._detect_mult:
                        return
            if not bfd_pkt.authenticate(sess._auth_keys):
                LOG.debug('[BFD][%s][AUTH] BFD Control authentication failed.', hex(sess._local_discr))
                return
        if sess is not None:
            if not sess._remote_addr_config:
                sess.set_remote_addr(eth.src, ip_pkt.src)
            sess.recv(bfd_pkt)