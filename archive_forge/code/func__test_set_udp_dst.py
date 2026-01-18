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
def _test_set_udp_dst(self, udp_dst):
    header = ofproto.OXM_OF_UDP_DST
    match = OFPMatch()
    match.set_udp_dst(udp_dst)
    self._test_serialize_and_parser(match, header, udp_dst)