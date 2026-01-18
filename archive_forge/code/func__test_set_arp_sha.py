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
def _test_set_arp_sha(self, arp_sha, mask=None):
    header = ofproto.OXM_OF_ARP_SHA
    match = OFPMatch()
    arp_sha = mac.haddr_to_bin(arp_sha)
    if mask is None:
        match.set_arp_sha(arp_sha)
    else:
        header = ofproto.OXM_OF_ARP_SHA_W
        mask = mac.haddr_to_bin(mask)
        match.set_arp_sha_masked(arp_sha, mask)
        arp_sha = mac.haddr_bitand(arp_sha, mask)
    self._test_serialize_and_parser(match, header, arp_sha, mask)