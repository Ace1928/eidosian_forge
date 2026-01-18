import collections
import functools
import inspect
from oslo_log import log as logging
from oslo_messaging import rpc
class MessageProcessor(object):
    queue = None

    def __init__(self, name):
        self.name = name

    def __call__(self):
        message = self.queue.get()
        if message is None:
            LOG.debug('[%s] No messages', self.name)
            return False
        try:
            method = getattr(self, message.name)
        except AttributeError:
            LOG.error('[%s] Bad message name "%s"' % (self.name, message.name))
            raise
        else:
            LOG.info('[%s] %r' % (self.name, message.data))
        method(message.data)
        return True

    @asynchronous
    def noop(self, count=1):
        """Insert <count> No-op operations in the message queue."""
        assert isinstance(count, int)
        if count > 1:
            self.queue.send_priority('noop', self.noop.MessageData(count - 1))

    @asynchronous
    def _execute(self, func):
        """Insert a function call in the message queue.

        The function takes no arguments, so use functools.partial to curry the
        arguments before passing it here.
        """
        func()

    def call(self, func, *args, **kwargs):
        """Insert a function call in the message queue."""
        self._execute(functools.partial(func, *args, **kwargs))

    def clear(self):
        """Delete all the messages from the queue."""
        self.queue.clear()