import time
import warnings
from typing import Tuple
from zmq import ETERM, POLLERR, POLLIN, POLLOUT, Poller, ZMQError
from .minitornado.ioloop import PeriodicCallback, PollIOLoop
from .minitornado.log import gen_log
class ZMQIOLoop(PollIOLoop):
    """ZMQ subclass of tornado's IOLoop

    Minor modifications, so that .current/.instance return self
    """
    _zmq_impl = ZMQPoller

    def initialize(self, impl=None, **kwargs):
        impl = self._zmq_impl() if impl is None else impl
        super().initialize(impl=impl, **kwargs)

    @classmethod
    def instance(cls, *args, **kwargs):
        """Returns a global `IOLoop` instance.

        Most applications have a single, global `IOLoop` running on the
        main thread.  Use this method to get this instance from
        another thread.  To get the current thread's `IOLoop`, use `current()`.
        """
        if tornado_version >= (3,):
            PollIOLoop.configure(cls)
        loop = PollIOLoop.instance(*args, **kwargs)
        if not isinstance(loop, cls):
            warnings.warn(f'IOLoop.current expected instance of {cls!r}, got {loop!r}', RuntimeWarning, stacklevel=2)
        return loop

    @classmethod
    def current(cls, *args, **kwargs):
        """Returns the current thread’s IOLoop."""
        if tornado_version >= (3,):
            PollIOLoop.configure(cls)
        loop = PollIOLoop.current(*args, **kwargs)
        if not isinstance(loop, cls):
            warnings.warn(f'IOLoop.current expected instance of {cls!r}, got {loop!r}', RuntimeWarning, stacklevel=2)
        return loop

    def start(self):
        try:
            super().start()
        except ZMQError as e:
            if e.errno == ETERM:
                pass
            else:
                raise