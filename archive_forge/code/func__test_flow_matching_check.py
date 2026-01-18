import binascii
import inspect
import json
import logging
import math
import netaddr
import os
import signal
import sys
import time
import traceback
from random import randint
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import hub
from os_ken.lib import stringify
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
def _test_flow_matching_check(self, pkt):
    self.logger.debug('egress:[%s]', packet.Packet(pkt.get(KEY_EGRESS)))
    self.logger.debug('packet_in:[%s]', packet.Packet(pkt.get(KEY_PKT_IN)))
    try:
        self._wait()
    except TestTimeout:
        return TIMEOUT
    assert len(self.rcv_msgs) == 1
    msg = self.rcv_msgs[0]
    assert msg.__class__.__name__ == 'OFPPacketIn'
    self.logger.debug('dpid=%s : receive_packet[%s]', dpid_lib.dpid_to_str(msg.datapath.id), packet.Packet(msg.data))
    pkt_in_src_model = self.tester_sw if KEY_EGRESS in pkt else self.target_sw
    model_pkt = pkt[KEY_EGRESS] if KEY_EGRESS in pkt else pkt[KEY_PKT_IN]
    if hasattr(msg.datapath.ofproto, 'OFPR_NO_MATCH'):
        invalid_packet_in_reason = [msg.datapath.ofproto.OFPR_NO_MATCH]
    else:
        invalid_packet_in_reason = [msg.datapath.ofproto.OFPR_TABLE_MISS]
    if hasattr(msg.datapath.ofproto, 'OFPR_INVALID_TTL'):
        invalid_packet_in_reason.append(msg.datapath.ofproto.OFPR_INVALID_TTL)
    if msg.datapath.id != pkt_in_src_model.dp.id:
        pkt_type = 'packet-in'
        err_msg = 'SW[dpid=%s]' % dpid_lib.dpid_to_str(msg.datapath.id)
    elif msg.reason in invalid_packet_in_reason:
        pkt_type = 'packet-in'
        err_msg = 'OFPPacketIn[reason=%d]' % msg.reason
    elif repr(msg.data) != repr(model_pkt):
        pkt_type = 'packet'
        err_msg = self._diff_packets(packet.Packet(model_pkt), packet.Packet(msg.data))
    else:
        return TEST_OK
    raise TestFailure(self.state, pkt_type=pkt_type, detail=err_msg)