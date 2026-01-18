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
def _test_set_ipv6_src(self, ipv6, mask=None):
    header = ofproto.OXM_OF_IPV6_SRC
    match = OFPMatch()
    ipv6 = [int(x, 16) for x in ipv6.split(':')]
    if mask is None:
        match.set_ipv6_src(ipv6)
    else:
        header = ofproto.OXM_OF_IPV6_SRC_W
        mask = [int(x, 16) for x in mask.split(':')]
        match.set_ipv6_src_masked(ipv6, mask)
        ipv6 = [x & y for x, y in zip(ipv6, mask)]
    self._test_serialize_and_parser(match, header, ipv6, mask)