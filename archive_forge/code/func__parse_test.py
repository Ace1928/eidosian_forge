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
@classmethod
def _parse_test(cls, buf):

    def __test_pkt_from_json(test):
        data = eval('/'.join(test))
        data.serialize()
        return bytes(data.data)
    target_dp = DummyDatapath(OfTester.target_ver)
    tester_dp = DummyDatapath(OfTester.tester_ver)
    description = buf.get(KEY_DESC)
    prerequisite = []
    if KEY_PREREQ not in buf:
        raise ValueError('a test requires a "%s" block' % KEY_PREREQ)
    for flow in buf[KEY_PREREQ]:
        msg = ofproto_parser.ofp_msg_from_jsondict(target_dp, flow)
        msg.serialize()
        prerequisite.append(msg)
    tests = []
    if KEY_TESTS not in buf:
        raise ValueError('a test requires a "%s" block.' % KEY_TESTS)
    for test in buf[KEY_TESTS]:
        if len(test) != 2:
            raise ValueError('"%s" block requires "%s" field and one of "%s" or "%s" or "%s" field.' % (KEY_TESTS, KEY_INGRESS, KEY_EGRESS, KEY_PKT_IN, KEY_TBL_MISS))
        test_pkt = {}
        if KEY_INGRESS not in test:
            raise ValueError('a test requires "%s" field.' % KEY_INGRESS)
        if isinstance(test[KEY_INGRESS], list):
            test_pkt[KEY_INGRESS] = __test_pkt_from_json(test[KEY_INGRESS])
        elif isinstance(test[KEY_INGRESS], dict):
            test_pkt[KEY_PACKETS] = {'packet_text': test[KEY_INGRESS][KEY_PACKETS][KEY_DATA], 'packet_binary': __test_pkt_from_json(test[KEY_INGRESS][KEY_PACKETS][KEY_DATA]), KEY_DURATION_TIME: test[KEY_INGRESS][KEY_PACKETS].get(KEY_DURATION_TIME, DEFAULT_DURATION_TIME), KEY_PKTPS: test[KEY_INGRESS][KEY_PACKETS].get(KEY_PKTPS, DEFAULT_PKTPS), 'randomize': True in [line.find('randint') != -1 for line in test[KEY_INGRESS][KEY_PACKETS][KEY_DATA]]}
        else:
            raise ValueError('invalid format: "%s" field' % KEY_INGRESS)
        if KEY_EGRESS in test:
            if isinstance(test[KEY_EGRESS], list):
                test_pkt[KEY_EGRESS] = __test_pkt_from_json(test[KEY_EGRESS])
            elif isinstance(test[KEY_EGRESS], dict):
                throughputs = []
                for throughput in test[KEY_EGRESS][KEY_THROUGHPUT]:
                    one = {}
                    mod = {'OFPFlowMod': {'cookie': THROUGHPUT_COOKIE, 'priority': THROUGHPUT_PRIORITY, 'match': {'OFPMatch': throughput[KEY_MATCH]}}}
                    msg = ofproto_parser.ofp_msg_from_jsondict(tester_dp, mod)
                    one[KEY_FLOW] = msg
                    one[KEY_KBPS] = throughput.get(KEY_KBPS)
                    one[KEY_PKTPS] = throughput.get(KEY_PKTPS)
                    if not bool(one[KEY_KBPS]) != bool(one[KEY_PKTPS]):
                        raise ValueError('"%s" requires either "%s" or "%s".' % (KEY_THROUGHPUT, KEY_KBPS, KEY_PKTPS))
                    throughputs.append(one)
                test_pkt[KEY_THROUGHPUT] = throughputs
            else:
                raise ValueError('invalid format: "%s" field' % KEY_EGRESS)
        elif KEY_PKT_IN in test:
            test_pkt[KEY_PKT_IN] = __test_pkt_from_json(test[KEY_PKT_IN])
        elif KEY_TBL_MISS in test:
            test_pkt[KEY_TBL_MISS] = test[KEY_TBL_MISS]
        tests.append(test_pkt)
    return (description, prerequisite, tests)