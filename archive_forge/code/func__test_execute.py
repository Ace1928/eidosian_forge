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
def _test_execute(self, test, description):
    if isinstance(self.target_sw.dp, DummyDatapath) or isinstance(self.tester_sw.dp, DummyDatapath):
        self.logger.info('waiting for switches connection...')
        self.sw_waiter = hub.Event()
        self.sw_waiter.wait()
        self.sw_waiter = None
    if description:
        self.logger.info('%s', description)
    self.thread_msg = None
    try:
        self._test(STATE_INIT_METER)
        self._test(STATE_INIT_GROUP)
        self._test(STATE_INIT_FLOW, self.target_sw)
        self._test(STATE_INIT_THROUGHPUT_FLOW, self.tester_sw)
        for flow in test.prerequisite:
            if isinstance(flow, self.target_sw.dp.ofproto_parser.OFPFlowMod):
                self._test(STATE_FLOW_INSTALL, self.target_sw, flow)
                self._test(STATE_FLOW_EXIST_CHK, self.target_sw.send_flow_stats, flow)
            elif isinstance(flow, self.target_sw.dp.ofproto_parser.OFPMeterMod):
                self._test(STATE_METER_INSTALL, self.target_sw, flow)
                self._test(STATE_METER_EXIST_CHK, self.target_sw.send_meter_config_stats, flow)
            elif isinstance(flow, self.target_sw.dp.ofproto_parser.OFPGroupMod):
                self._test(STATE_GROUP_INSTALL, self.target_sw, flow)
                self._test(STATE_GROUP_EXIST_CHK, self.target_sw.send_group_desc_stats, flow)
        for pkt in test.tests:
            if KEY_EGRESS in pkt or KEY_PKT_IN in pkt:
                target_pkt_count = [self._test(STATE_TARGET_PKT_COUNT, True)]
                tester_pkt_count = [self._test(STATE_TESTER_PKT_COUNT, False)]
            elif KEY_THROUGHPUT in pkt:
                for throughput in pkt[KEY_THROUGHPUT]:
                    flow = throughput[KEY_FLOW]
                    self._test(STATE_THROUGHPUT_FLOW_INSTALL, self.tester_sw, flow)
                    self._test(STATE_THROUGHPUT_FLOW_EXIST_CHK, self.tester_sw.send_flow_stats, flow)
                start = self._test(STATE_GET_THROUGHPUT)
            elif KEY_TBL_MISS in pkt:
                before_stats = self._test(STATE_GET_MATCH_COUNT)
            if KEY_INGRESS in pkt:
                self._one_time_packet_send(pkt)
            elif KEY_PACKETS in pkt:
                self._continuous_packet_send(pkt)
            if KEY_EGRESS in pkt or KEY_PKT_IN in pkt:
                result = self._test(STATE_FLOW_MATCH_CHK, pkt)
                if result == TIMEOUT:
                    target_pkt_count.append(self._test(STATE_TARGET_PKT_COUNT, True))
                    tester_pkt_count.append(self._test(STATE_TESTER_PKT_COUNT, False))
                    test_type = KEY_EGRESS if KEY_EGRESS in pkt else KEY_PKT_IN
                    self._test(STATE_NO_PKTIN_REASON, test_type, target_pkt_count, tester_pkt_count)
            elif KEY_THROUGHPUT in pkt:
                end = self._test(STATE_GET_THROUGHPUT)
                self._test(STATE_THROUGHPUT_CHK, pkt[KEY_THROUGHPUT], start, end)
            elif KEY_TBL_MISS in pkt:
                self._test(STATE_SEND_BARRIER)
                hub.sleep(INTERVAL)
                self._test(STATE_FLOW_UNMATCH_CHK, before_stats, pkt)
        result = [TEST_OK]
        result_type = TEST_OK
    except (TestFailure, TestError, TestTimeout, TestReceiveError) as err:
        result = [TEST_ERROR, str(err)]
        result_type = str(err).split(':', 1)[0]
    finally:
        self.ingress_event = None
        for tid in self.ingress_threads:
            hub.kill(tid)
        self.ingress_threads = []
    self.logger.info('    %-100s %s', test.description, result[0])
    if 1 < len(result):
        self.logger.info('        %s', result[1])
        if result[1] == OSKEN_INTERNAL_ERROR or result == 'An unknown exception':
            self.logger.error(traceback.format_exc())
    hub.sleep(0)
    return result_type