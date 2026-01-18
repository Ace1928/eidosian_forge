import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
class TestOFPPort(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPPort
    """

    def test_init(self):
        port_no = 1119692796
        hw_addr = 'c0:26:53:c4:29:e2'
        name = b'name'.ljust(16)
        config = 2226555987
        state = 1678244809
        curr = 2850556459
        advertised = 2025421682
        supported = 2120575149
        peer = 2757463021
        curr_speed = 2641353507
        max_speed = 1797291672
        fmt = ofproto.OFP_PORT_PACK_STR
        c = OFPPort(port_no, hw_addr, name, config, state, curr, advertised, supported, peer, curr_speed, max_speed)
        self.assertEqual(port_no, c.port_no)
        self.assertEqual(hw_addr, c.hw_addr)
        self.assertEqual(name, c.name)
        self.assertEqual(config, c.config)
        self.assertEqual(state, c.state)
        self.assertEqual(curr, c.curr)
        self.assertEqual(advertised, c.advertised)
        self.assertEqual(supported, c.supported)
        self.assertEqual(peer, c.peer)
        self.assertEqual(curr_speed, c.curr_speed)
        self.assertEqual(max_speed, c.max_speed)

    def _test_parser(self, port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed):
        name = b'name'.ljust(16)
        fmt = ofproto.OFP_PORT_PACK_STR
        buf = pack(fmt, port_no, addrconv.mac.text_to_bin(hw_addr), name, config, state, curr, advertised, supported, peer, curr_speed, max_speed)
        res = OFPPort.parser(buf, 0)
        self.assertEqual(port_no, res.port_no)
        self.assertEqual(hw_addr, res.hw_addr)
        self.assertEqual(name, res.name)
        self.assertEqual(config, res.config)
        self.assertEqual(state, res.state)
        self.assertEqual(curr, res.curr)
        self.assertEqual(advertised, res.advertised)
        self.assertEqual(supported, res.supported)
        self.assertEqual(peer, res.peer)
        self.assertEqual(curr_speed, res.curr_speed)
        self.assertEqual(max_speed, res.max_speed)

    def test_parser_mid(self):
        port_no = 1119692796
        hw_addr = 'c0:26:53:c4:29:e2'
        config = 2226555987
        state = 1678244809
        curr = 2850556459
        advertised = 2025421682
        supported = 2120575149
        peer = 2757463021
        curr_speed = 2641353507
        max_speed = 1797291672
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_max(self):
        port_no = ofproto.OFPP_ANY
        hw_addr = 'ff:ff:ff:ff:ff:ff'
        config = 4294967295
        state = 4294967295
        curr = 4294967295
        advertised = 4294967295
        supported = 4294967295
        peer = 4294967295
        curr_speed = 4294967295
        max_speed = 4294967295
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_min(self):
        port_no = 0
        hw_addr = '00:00:00:00:00:00'
        config = 0
        state = 0
        curr = 0
        advertised = 0
        supported = 0
        peer = 0
        curr_speed = 0
        max_speed = 0
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p1(self):
        port_no = ofproto.OFPP_MAX
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_PORT_DOWN
        state = ofproto.OFPPS_LINK_DOWN
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_10MB_HD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p2(self):
        port_no = ofproto.OFPP_IN_PORT
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_RECV
        state = ofproto.OFPPS_BLOCKED
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_10MB_FD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p3(self):
        port_no = ofproto.OFPP_TABLE
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_FWD
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_100MB_HD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p4(self):
        port_no = ofproto.OFPP_NORMAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_100MB_FD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p5(self):
        port_no = ofproto.OFPP_FLOOD
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_1GB_HD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p6(self):
        port_no = ofproto.OFPP_ALL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_1GB_FD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p7(self):
        port_no = ofproto.OFPP_CONTROLLER
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_10GB_FD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p8(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_40GB_FD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p9(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_100GB_FD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p10(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_1TB_FD
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p11(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_OTHER
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p12(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_COPPER
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p13(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_FIBER
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p14(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_AUTONEG
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p15(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_PAUSE
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p16(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = 'c0:26:53:c4:29:e2'
        config = ofproto.OFPPC_NO_PACKET_IN
        state = ofproto.OFPPS_LIVE
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_PAUSE_ASYM
        self._test_parser(port_no, hw_addr, config, state, curr, advertised, supported, peer, curr_speed, max_speed)