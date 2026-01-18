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
class OpenFlowSw(object):

    def __init__(self, dp, logger):
        super(OpenFlowSw, self).__init__()
        self.dp = dp
        self.logger = logger
        self.tester_send_port = CONF['test-switch']['tester_send_port']

    def send_msg(self, msg):
        if isinstance(self.dp, DummyDatapath):
            raise TestError(STATE_DISCONNECTED)
        msg.xid = None
        self.dp.set_xid(msg)
        self.dp.send_msg(msg)
        return msg.xid

    def add_flow(self, in_port=None, out_port=None):
        """ Add flow. """
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        match = parser.OFPMatch(in_port=in_port)
        actions = [parser.OFPActionOutput(out_port)]
        if ofp.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            mod = parser.OFPFlowMod(self.dp, match=match, cookie=0, command=ofp.OFPFC_ADD, actions=actions)
        else:
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(self.dp, cookie=0, command=ofp.OFPFC_ADD, match=match, instructions=inst)
        return self.send_msg(mod)

    def del_flows(self, cookie=0):
        """
        Delete all flow except default flow by using the cookie value.

        Note: In OpenFlow 1.0, DELETE and DELETE_STRICT commands can
        not be filtered by the cookie value and this value is ignored.
        """
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        cookie_mask = 0
        if cookie:
            cookie_mask = 18446744073709551615
        if ofp.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            match = parser.OFPMatch()
            mod = parser.OFPFlowMod(self.dp, match, cookie, ofp.OFPFC_DELETE)
        else:
            mod = parser.OFPFlowMod(self.dp, cookie=cookie, cookie_mask=cookie_mask, table_id=ofp.OFPTT_ALL, command=ofp.OFPFC_DELETE, out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY)
        return self.send_msg(mod)

    def del_meters(self):
        """ Delete all meter entries. """
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        if ofp.OFP_VERSION < ofproto_v1_3.OFP_VERSION:
            return None
        mod = parser.OFPMeterMod(self.dp, command=ofp.OFPMC_DELETE, flags=0, meter_id=ofp.OFPM_ALL)
        return self.send_msg(mod)

    def del_groups(self):
        """ Delete all group entries. """
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        if ofp.OFP_VERSION < ofproto_v1_2.OFP_VERSION:
            return None
        mod = parser.OFPGroupMod(self.dp, command=ofp.OFPGC_DELETE, type_=0, group_id=ofp.OFPG_ALL)
        return self.send_msg(mod)

    def send_barrier_request(self):
        """ send a BARRIER_REQUEST message."""
        parser = self.dp.ofproto_parser
        req = parser.OFPBarrierRequest(self.dp)
        return self.send_msg(req)

    def send_port_stats(self):
        """ Get port stats."""
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        flags = 0
        if ofp.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            port = ofp.OFPP_NONE
        else:
            port = ofp.OFPP_ANY
        req = parser.OFPPortStatsRequest(self.dp, flags, port)
        return self.send_msg(req)

    def send_flow_stats(self):
        """ Get all flow. """
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        if ofp.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            req = parser.OFPFlowStatsRequest(self.dp, 0, parser.OFPMatch(), 255, ofp.OFPP_NONE)
        else:
            req = parser.OFPFlowStatsRequest(self.dp, 0, ofp.OFPTT_ALL, ofp.OFPP_ANY, ofp.OFPG_ANY, 0, 0, parser.OFPMatch())
        return self.send_msg(req)

    def send_meter_config_stats(self):
        """ Get all meter. """
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        if ofp.OFP_VERSION < ofproto_v1_3.OFP_VERSION:
            return None
        stats = parser.OFPMeterConfigStatsRequest(self.dp)
        return self.send_msg(stats)

    def send_group_desc_stats(self):
        """ Get all group. """
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        if ofp.OFP_VERSION < ofproto_v1_2.OFP_VERSION:
            return None
        stats = parser.OFPGroupDescStatsRequest(self.dp)
        return self.send_msg(stats)

    def send_table_stats(self):
        """ Get table stats. """
        parser = self.dp.ofproto_parser
        req = parser.OFPTableStatsRequest(self.dp, 0)
        return self.send_msg(req)

    def send_packet_out(self, data):
        """ send a PacketOut message."""
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        actions = [parser.OFPActionOutput(self.tester_send_port)]
        out = parser.OFPPacketOut(datapath=self.dp, buffer_id=ofp.OFP_NO_BUFFER, data=data, in_port=ofp.OFPP_CONTROLLER, actions=actions)
        return self.send_msg(out)