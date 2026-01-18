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
def __get_version(opt):
    vers = {'openflow10': ofproto_v1_0.OFP_VERSION, 'openflow13': ofproto_v1_3.OFP_VERSION, 'openflow14': ofproto_v1_4.OFP_VERSION, 'openflow15': ofproto_v1_5.OFP_VERSION}
    ver = vers.get(opt.lower())
    if ver is None:
        self.logger.error('%s is not supported. Supported versions are %s.', opt, list(vers.keys()))
        self._test_end()
    return ver