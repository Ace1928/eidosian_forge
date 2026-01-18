import abc
import threading
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_messaging import exceptions
class PollStyleListenerAdapter(Listener):
    """A Listener that uses a PollStyleListener for message transfer. A
    dedicated thread is created to do message polling.
    """

    def __init__(self, poll_style_listener, batch_size, batch_timeout):
        super(PollStyleListenerAdapter, self).__init__(batch_size, batch_timeout, poll_style_listener.prefetch_size)
        self._poll_style_listener = poll_style_listener
        self._listen_thread = threading.Thread(target=self._runner)
        self._listen_thread.daemon = True
        self._started = False

    def start(self, on_incoming_callback):
        super(PollStyleListenerAdapter, self).start(on_incoming_callback)
        self._started = True
        self._listen_thread.start()

    @excutils.forever_retry_uncaught_exceptions
    def _runner(self):
        while self._started:
            incoming = self._poll_style_listener.poll(batch_size=self.batch_size, batch_timeout=self.batch_timeout)
            if incoming:
                self.on_incoming_callback(incoming)
        while True:
            incoming = self._poll_style_listener.poll(batch_size=self.batch_size, batch_timeout=self.batch_timeout)
            if not incoming:
                return
            self.on_incoming_callback(incoming)

    def stop(self):
        self._started = False
        self._poll_style_listener.stop()
        self._listen_thread.join()
        super(PollStyleListenerAdapter, self).stop()

    def cleanup(self):
        self._poll_style_listener.cleanup()