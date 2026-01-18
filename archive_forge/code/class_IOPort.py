import random
import threading
import time
from .messages import Message
from .parser import Parser
class IOPort(BaseIOPort):
    """Input / output port.

    This is a convenient wrapper around an input port and an output
    port which provides the functionality of both. Every method call
    is forwarded to the appropriate port.
    """
    _locking = False

    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.name = f'{str(input.name)} + {str(output.name)}'
        self._messages = self.input._messages
        self.closed = False
        self._lock = DummyLock()

    def _close(self):
        self.input.close()
        self.output.close()

    def _send(self, message):
        self.output.send(message)

    def _receive(self, block=True):
        return self.input.receive(block=block)