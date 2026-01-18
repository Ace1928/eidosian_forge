import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_3_parser import *
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_protocol
def _test_set_vlan_vid(self, vid, mask=None):
    header = ofproto.OXM_OF_VLAN_VID
    match = OFPMatch()
    if mask is None:
        match.set_vlan_vid(vid)
    else:
        header = ofproto.OXM_OF_VLAN_VID_W
        match.set_vlan_vid_masked(vid, mask)
    self._test_serialize_and_parser(match, header, vid, mask)