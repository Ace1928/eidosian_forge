import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
@staticmethod
def _is_error(msg):
    return ofp_event.ofp_msg_to_ev_cls(type(msg)) == ofp_event.EventOFPErrorMsg