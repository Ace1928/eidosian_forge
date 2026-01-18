from collections import deque
from threading import Event
from kombu.common import ignore_errors
from kombu.utils.encoding import bytes_to_str
from kombu.utils.imports import symbol_by_name
from .utils.graph import DependencyGraph, GraphFormatter
from .utils.imports import instantiate, qualname
from .utils.log import get_logger
class ConsumerStep(StartStopStep):
    """Bootstep that starts a message consumer."""
    requires = ('celery.worker.consumer:Connection',)
    consumers = None

    def get_consumers(self, channel):
        raise NotImplementedError('missing get_consumers')

    def start(self, c):
        channel = c.connection.channel()
        self.consumers = self.get_consumers(channel)
        for consumer in self.consumers or []:
            consumer.consume()

    def stop(self, c):
        self._close(c, True)

    def shutdown(self, c):
        self._close(c, False)

    def _close(self, c, cancel_consumers=True):
        channels = set()
        for consumer in self.consumers or []:
            if cancel_consumers:
                ignore_errors(c.connection, consumer.cancel)
            if consumer.channel:
                channels.add(consumer.channel)
        for channel in channels:
            ignore_errors(c.connection, channel.close)