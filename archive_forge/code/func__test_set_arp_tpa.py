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
def _test_set_arp_tpa(self, ip, mask=None):
    header = ofproto.OXM_OF_ARP_TPA
    match = OFPMatch()
    ip = unpack('!I', socket.inet_aton(ip))[0]
    if mask is None:
        match.set_arp_tpa(ip)
    else:
        mask = unpack('!I', socket.inet_aton(mask))[0]
        if mask + 1 >> 32 != 1:
            header = ofproto.OXM_OF_ARP_TPA_W
        match.set_arp_tpa_masked(ip, mask)
    self._test_serialize_and_parser(match, header, ip, mask)