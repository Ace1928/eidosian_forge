import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def _get_obj(self, actions=None):

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    c = OFPFlowMod(Datapath, self.match, self.cookie['val'], self.command['val'], self.idle_timeout['val'], self.hard_timeout['val'], self.priority['val'], self.buffer_id['val'], self.out_port['val'], self.flags['val'], actions)
    return c