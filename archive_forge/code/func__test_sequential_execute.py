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
def _test_sequential_execute(self, test_dir):
    """ Execute OpenFlow Switch test. """
    tests = TestPatterns(test_dir, self.logger)
    if not tests:
        self.logger.warning(NO_TEST_FILE)
        self._test_end()
    test_report = {}
    self.logger.info('--- Test start ---')
    test_keys = list(tests.keys())
    test_keys.sort()
    for file_name in test_keys:
        report = self._test_file_execute(tests[file_name])
        for result, descriptions in report.items():
            test_report.setdefault(result, [])
            test_report[result].extend(descriptions)
    self._test_end(msg='---  Test end  ---', report=test_report)