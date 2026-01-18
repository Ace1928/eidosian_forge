import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
class _SwitchInfo(object):

    def __init__(self, datapath):
        self.datapath = datapath
        self.xids = {}
        self.barriers = {}
        self.results = {}