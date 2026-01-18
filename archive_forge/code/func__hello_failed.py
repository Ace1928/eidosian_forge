import itertools
import logging
import warnings
import os_ken.base.app_manager
from os_ken.lib import hub
from os_ken import utils
from os_ken.controller import ofp_event
from os_ken.controller.controller import OpenFlowController
from os_ken.controller.handler import set_ev_handler
from os_ken.controller.handler import HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER,\
from os_ken.ofproto import ofproto_parser
def _hello_failed(self, datapath, error_desc):
    self.logger.error('%s on datapath %s', error_desc, datapath.address)
    error_msg = datapath.ofproto_parser.OFPErrorMsg(datapath=datapath, type_=datapath.ofproto.OFPET_HELLO_FAILED, code=datapath.ofproto.OFPHFC_INCOMPATIBLE, data=error_desc)
    datapath.send_msg(error_msg, close_socket=True)