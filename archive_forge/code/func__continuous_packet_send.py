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
def _continuous_packet_send(self, pkt):
    assert self.ingress_event is None
    pkt_text = pkt[KEY_PACKETS]['packet_text']
    pkt_bin = pkt[KEY_PACKETS]['packet_binary']
    pktps = pkt[KEY_PACKETS][KEY_PKTPS]
    duration_time = pkt[KEY_PACKETS][KEY_DURATION_TIME]
    randomize = pkt[KEY_PACKETS]['randomize']
    self.logger.debug('send_packet:[%s]', packet.Packet(pkt_bin))
    self.logger.debug('pktps:[%d]', pktps)
    self.logger.debug('duration_time:[%d]', duration_time)
    arg = {'packet_text': pkt_text, 'packet_binary': pkt_bin, 'thread_counter': 0, 'dot_span': int(CONTINUOUS_PROGRESS_SPAN / CONTINUOUS_THREAD_INTVL), 'packet_counter': float(0), 'packet_counter_inc': pktps * CONTINUOUS_THREAD_INTVL, 'randomize': randomize}
    try:
        self.ingress_event = hub.Event()
        tid = hub.spawn(self._send_packet_thread, arg)
        self.ingress_threads.append(tid)
        self.ingress_event.wait(duration_time)
        if self.thread_msg is not None:
            raise self.thread_msg
    finally:
        sys.stdout.write('\r\n')
        sys.stdout.flush()