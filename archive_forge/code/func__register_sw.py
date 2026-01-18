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
def _register_sw(self, dp):
    vers = {ofproto_v1_0.OFP_VERSION: 'openflow10', ofproto_v1_3.OFP_VERSION: 'openflow13', ofproto_v1_4.OFP_VERSION: 'openflow14', ofproto_v1_5.OFP_VERSION: 'openflow15'}
    if dp.id == self.target_dpid:
        if dp.ofproto.OFP_VERSION != OfTester.target_ver:
            msg = 'Join target SW, but ofp version is not %s.' % vers[OfTester.target_ver]
        else:
            self.target_sw.dp = dp
            msg = 'Join target SW.'
    elif dp.id == self.tester_dpid:
        if dp.ofproto.OFP_VERSION != OfTester.tester_ver:
            msg = 'Join tester SW, but ofp version is not %s.' % vers[OfTester.tester_ver]
        else:
            self.tester_sw.dp = dp
            msg = 'Join tester SW.'
    else:
        msg = 'Connect unknown SW.'
    if dp.id:
        self.logger.info('dpid=%s : %s', dpid_lib.dpid_to_str(dp.id), msg)
    if not (isinstance(self.target_sw.dp, DummyDatapath) or isinstance(self.tester_sw.dp, DummyDatapath)):
        if self.sw_waiter is not None:
            self.sw_waiter.set()