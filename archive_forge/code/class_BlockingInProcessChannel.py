import sys
from queue import Empty, Queue
from traitlets import Type
from .channels import InProcessChannel
from .client import InProcessKernelClient
class BlockingInProcessChannel(InProcessChannel):
    """A blocking in-process channel."""

    def __init__(self, *args, **kwds):
        """Initialize the channel."""
        super().__init__(*args, **kwds)
        self._in_queue: Queue[object] = Queue()

    def call_handlers(self, msg):
        """Call the handlers for a message."""
        self._in_queue.put(msg)

    def get_msg(self, block=True, timeout=None):
        """Gets a message if there is one that is ready."""
        if timeout is None:
            timeout = 604800
        return self._in_queue.get(block, timeout)

    def get_msgs(self):
        """Get all messages that are currently ready."""
        msgs = []
        while True:
            try:
                msgs.append(self.get_msg(block=False))
            except Empty:
                break
        return msgs

    def msg_ready(self):
        """Is there a message that has been received?"""
        return not self._in_queue.empty()