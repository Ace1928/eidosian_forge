import time
import warnings
from typing import Tuple
from zmq import ETERM, POLLERR, POLLIN, POLLOUT, Poller, ZMQError
from .minitornado.ioloop import PeriodicCallback, PollIOLoop
from .minitornado.log import gen_log
@staticmethod
def _remap_events(z_events):
    """translate zmq.POLLIN/OUT/ERR event masks into IOLoop.READ/WRITE/ERROR"""
    events = 0
    if z_events & POLLIN:
        events |= IOLoop.READ
    if z_events & POLLOUT:
        events |= IOLoop.WRITE
    if z_events & POLLERR:
        events |= IOLoop.ERROR
    return events