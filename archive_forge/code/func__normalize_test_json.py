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
def _normalize_test_json(self, val):

    def __replace_port_name(k, v):
        for port_name in ['target_recv_port', 'target_send_port_1', 'target_send_port_2', 'tester_send_port', 'tester_recv_port_1', 'tester_recv_port_2']:
            if v[k] == port_name:
                v[k] = CONF['test-switch'][port_name]
    if isinstance(val, dict):
        for k, v in val.items():
            if k == 'OFPActionOutput':
                if 'port' in v:
                    __replace_port_name('port', v)
            elif k == 'OXMTlv':
                if v.get('field', '') == 'in_port':
                    __replace_port_name('value', v)
            self._normalize_test_json(v)
    elif isinstance(val, list):
        for v in val:
            self._normalize_test_json(v)