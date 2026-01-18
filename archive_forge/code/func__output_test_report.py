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
def _output_test_report(self, report):
    self.logger.info('%s--- Test report ---', os.linesep)
    error_count = 0
    for result_type in sorted(list(report.keys())):
        test_descriptions = report[result_type]
        if result_type == TEST_OK:
            continue
        error_count += len(test_descriptions)
        self.logger.info('%s(%d)', result_type, len(test_descriptions))
        for file_desc, test_desc in test_descriptions:
            self.logger.info('    %-40s %s', file_desc, test_desc)
    self.logger.info('%s%s(%d) / %s(%d)', os.linesep, TEST_OK, len(report.get(TEST_OK, [])), TEST_ERROR, error_count)