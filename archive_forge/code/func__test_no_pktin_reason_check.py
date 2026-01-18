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
def _test_no_pktin_reason_check(self, test_type, target_pkt_count, tester_pkt_count):
    before_target_receive = target_pkt_count[0][self.target_recv_port]['rx']
    before_target_send = target_pkt_count[0][self.target_send_port_1]['tx']
    before_tester_receive = tester_pkt_count[0][self.tester_recv_port_1]['rx']
    before_tester_send = tester_pkt_count[0][self.tester_send_port]['tx']
    after_target_receive = target_pkt_count[1][self.target_recv_port]['rx']
    after_target_send = target_pkt_count[1][self.target_send_port_1]['tx']
    after_tester_receive = tester_pkt_count[1][self.tester_recv_port_1]['rx']
    after_tester_send = tester_pkt_count[1][self.tester_send_port]['tx']
    if after_tester_send == before_tester_send:
        log_msg = 'no change in tx_packets on tester.'
    elif after_target_receive == before_target_receive:
        log_msg = 'no change in rx_packets on target.'
    elif test_type == KEY_EGRESS:
        if after_target_send == before_target_send:
            log_msg = 'no change in tx_packets on target.'
        elif after_tester_receive == before_tester_receive:
            log_msg = 'no change in rx_packets on tester.'
        else:
            log_msg = 'increment in rx_packets in tester.'
    else:
        assert test_type == KEY_PKT_IN
        log_msg = 'no packet-in.'
    raise TestFailure(self.state, detail=log_msg)