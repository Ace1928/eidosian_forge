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
def _test_throughput_check(self, throughputs, start, end):
    msgs = []
    elapsed_sec = end[0] - start[0]
    for throughput in throughputs:
        match = str(throughput[KEY_FLOW].match)
        fields = dict(throughput[KEY_FLOW].match._fields2)
        if match not in start[1] or match not in end[1]:
            raise TestError(self.state, match=match)
        increased_bytes = end[1][match][0] - start[1][match][0]
        increased_packets = end[1][match][1] - start[1][match][1]
        if throughput[KEY_PKTPS]:
            key = KEY_PKTPS
            conv = 1
            measured_value = increased_packets
            unit = 'pktps'
        elif throughput[KEY_KBPS]:
            key = KEY_KBPS
            conv = 1024 / 8
            measured_value = increased_bytes
            unit = 'kbps'
        else:
            raise OSKenException('An invalid key exists that is neither "%s" nor "%s".' % (KEY_KBPS, KEY_PKTPS))
        expected_value = throughput[key] * elapsed_sec * conv
        margin = expected_value * THROUGHPUT_THRESHOLD
        self.logger.debug('measured_value:[%s]', measured_value)
        self.logger.debug('expected_value:[%s]', expected_value)
        self.logger.debug('margin:[%s]', margin)
        if math.fabs(measured_value - expected_value) > margin:
            msgs.append('{0} {1:.2f}{2}'.format(fields, measured_value / elapsed_sec / conv, unit))
    if msgs:
        raise TestFailure(self.state, detail=', '.join(msgs))