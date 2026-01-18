import inspect
import time
from os_ken.controller import handler
from os_ken import ofproto
from . import event
def _create_ofp_msg_ev_from_module(ofp_parser):
    for _k, cls in inspect.getmembers(ofp_parser, inspect.isclass):
        if not hasattr(cls, 'cls_msg_type'):
            continue
        _create_ofp_msg_ev_class(cls)