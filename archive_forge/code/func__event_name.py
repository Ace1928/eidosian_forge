import inspect
import logging
from os_ken import utils
from os_ken.controller import event
from os_ken.lib.packet import zebra
def _event_name(body_cls):
    return 'Event%s' % body_cls.__name__