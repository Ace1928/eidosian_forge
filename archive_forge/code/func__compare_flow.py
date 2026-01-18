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
def _compare_flow(self, stats1, stats2):

    def __reasm_match(match):
        """ reassemble match_fields. """
        match_fields = match.to_jsondict()
        match_fields['OFPMatch'].pop('wildcards', None)
        return match_fields
    attr_list = ['cookie', 'priority', 'hard_timeout', 'idle_timeout', 'match']
    if self.target_sw.dp.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        attr_list += ['actions']
    else:
        attr_list += ['table_id', 'instructions']
    for attr in attr_list:
        value1 = getattr(stats1, attr)
        value2 = getattr(stats2, attr)
        if attr in ['actions', 'instructions']:
            value1 = sorted(value1, key=lambda x: x.type)
            value2 = sorted(value2, key=lambda x: x.type)
        elif attr == 'match':
            value1 = __reasm_match(value1)
            value2 = __reasm_match(value2)
        if str(value1) != str(value2):
            return (False, 'flow_stats(%s != %s)' % (value1, value2))
    return (True, None)