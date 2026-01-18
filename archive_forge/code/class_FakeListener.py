import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
class FakeListener(base.PollStyleListener):

    def __init__(self, exchange_manager, targets, pool=None):
        super(FakeListener, self).__init__()
        self._exchange_manager = exchange_manager
        self._targets = targets
        self._pool = pool
        self._stopped = eventletutils.Event()
        for target in self._targets:
            exchange = self._exchange_manager.get_exchange(target.exchange)
            exchange.ensure_queue(target, pool)

    @base.batch_poll_helper
    def poll(self, timeout=None):
        if timeout is not None:
            deadline = time.time() + timeout
        else:
            deadline = None
        while not self._stopped.is_set():
            for target in self._targets:
                exchange = self._exchange_manager.get_exchange(target.exchange)
                ctxt, message, reply_q, requeue = exchange.poll(target, self._pool)
                if message is not None:
                    message = FakeIncomingMessage(ctxt, message, reply_q, requeue)
                    return message
            if deadline is not None:
                pause = deadline - time.time()
                if pause < 0:
                    break
                pause = min(pause, 0.05)
            else:
                pause = 0.05
            time.sleep(pause)
        return None

    def stop(self):
        self._stopped.set()