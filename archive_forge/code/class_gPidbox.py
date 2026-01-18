import socket
import threading
from kombu.common import ignore_errors
from kombu.utils.encoding import safe_str
from celery.utils.collections import AttributeDict
from celery.utils.functional import pass1
from celery.utils.log import get_logger
from . import control
class gPidbox(Pidbox):
    """Worker pidbox (greenlet)."""
    _node_shutdown = None
    _node_stopped = None
    _resets = 0

    def start(self, c):
        c.pool.spawn_n(self.loop, c)

    def on_stop(self):
        if self._node_stopped:
            self._node_shutdown.set()
            debug('Waiting for broadcast thread to shutdown...')
            self._node_stopped.wait()
            self._node_stopped = self._node_shutdown = None

    def reset(self):
        self._resets += 1

    def _do_reset(self, c, connection):
        self._close_channel(c)
        self.node.channel = connection.channel()
        self.consumer = self.node.listen(callback=self.on_message)
        self.consumer.consume()

    def loop(self, c):
        resets = [self._resets]
        shutdown = self._node_shutdown = threading.Event()
        stopped = self._node_stopped = threading.Event()
        try:
            with c.connection_for_read() as connection:
                info('pidbox: Connected to %s.', connection.as_uri())
                self._do_reset(c, connection)
                while not shutdown.is_set() and c.connection:
                    if resets[0] < self._resets:
                        resets[0] += 1
                        self._do_reset(c, connection)
                    try:
                        connection.drain_events(timeout=1.0)
                    except socket.timeout:
                        pass
        finally:
            stopped.set()