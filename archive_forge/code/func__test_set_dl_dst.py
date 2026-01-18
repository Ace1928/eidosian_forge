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
def _test_set_dl_dst(self, dl_dst, mask=None):
    header = ofproto.OXM_OF_ETH_DST
    match = OFPMatch()
    dl_dst = mac.haddr_to_bin(dl_dst)
    if mask is None:
        match.set_dl_dst(dl_dst)
    else:
        header = ofproto.OXM_OF_ETH_DST_W
        mask = mac.haddr_to_bin(mask)
        match.set_dl_dst_masked(dl_dst, mask)
        dl_dst = mac.haddr_bitand(dl_dst, mask)
    self._test_serialize_and_parser(match, header, dl_dst, mask)