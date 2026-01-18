import time
import warnings
from typing import Tuple
from zmq import ETERM, POLLERR, POLLIN, POLLOUT, Poller, ZMQError
from .minitornado.ioloop import PeriodicCallback, PollIOLoop
from .minitornado.log import gen_log
class ZMQPoller:
    """A poller that can be used in the tornado IOLoop.

    This simply wraps a regular zmq.Poller, scaling the timeout
    by 1000, so that it is in seconds rather than milliseconds.
    """

    def __init__(self):
        self._poller = Poller()

    @staticmethod
    def _map_events(events):
        """translate IOLoop.READ/WRITE/ERROR event masks into zmq.POLLIN/OUT/ERR"""
        z_events = 0
        if events & IOLoop.READ:
            z_events |= POLLIN
        if events & IOLoop.WRITE:
            z_events |= POLLOUT
        if events & IOLoop.ERROR:
            z_events |= POLLERR
        return z_events

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

    def register(self, fd, events):
        return self._poller.register(fd, self._map_events(events))

    def modify(self, fd, events):
        return self._poller.modify(fd, self._map_events(events))

    def unregister(self, fd):
        return self._poller.unregister(fd)

    def poll(self, timeout):
        """poll in seconds rather than milliseconds.

        Event masks will be IOLoop.READ/WRITE/ERROR
        """
        z_events = self._poller.poll(1000 * timeout)
        return [(fd, self._remap_events(evt)) for fd, evt in z_events]

    def close(self):
        pass