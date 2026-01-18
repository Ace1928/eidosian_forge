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
def _diff_packets(cls, model_pkt, rcv_pkt):
    msg = []
    for rcv_p in rcv_pkt.protocols:
        if not isinstance(rcv_p, bytes):
            model_protocols = model_pkt.get_protocols(type(rcv_p))
            if len(model_protocols) == 1:
                model_p = model_protocols[0]
                diff = []
                for attr in rcv_p.__dict__:
                    if attr.startswith('_'):
                        continue
                    if callable(attr):
                        continue
                    if hasattr(rcv_p.__class__, attr):
                        continue
                    rcv_attr = repr(getattr(rcv_p, attr))
                    model_attr = repr(getattr(model_p, attr))
                    if rcv_attr != model_attr:
                        diff.append('%s=%s' % (attr, rcv_attr))
                if diff:
                    msg.append('%s(%s)' % (rcv_p.__class__.__name__, ','.join(diff)))
            elif not model_protocols or not str(rcv_p) in str(model_protocols):
                msg.append(str(rcv_p))
        else:
            model_p = ''
            for p in model_pkt.protocols:
                if isinstance(p, bytes):
                    model_p = p
                    break
            if model_p != rcv_p:
                msg.append('str(%s)' % repr(rcv_p))
    if msg:
        return '/'.join(msg)
    else:
        return 'Encounter an error during packet comparison. it is malformed.'