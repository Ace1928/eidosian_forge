import random
import threading
import time
from .messages import Message
from .parser import Parser
class BaseOutput(BasePort):
    """
    Base class for output port.

    Subclass and override _send() to create a new port type.  (See
    portmidi.py for how to do this.)
    """
    is_output = True

    def __init__(self, name='', autoreset=False, **kwargs):
        """Create an output port

        name is the port name, as returned by output_names(). If
        name is not passed, the default output is used instead.
        """
        BasePort.__init__(self, name, **kwargs)
        self.autoreset = autoreset

    def _send(self, msg):
        pass

    def send(self, msg):
        """Send a message on the port.

        A copy of the message will be sent, so you can safely modify
        the original message without any unexpected consequences.
        """
        if not self.is_output:
            raise ValueError('Not an output port')
        elif not isinstance(msg, Message):
            raise TypeError('argument to send() must be a Message')
        elif self.closed:
            raise ValueError('send() called on closed port')
        with self._lock:
            self._send(msg.copy())

    def reset(self):
        """Send "All Notes Off" and "Reset All Controllers" on all channels"""
        if self.closed:
            return
        for msg in reset_messages():
            self.send(msg)

    def panic(self):
        """Send "All Sounds Off" on all channels.

        This will mute all sounding notes regardless of
        envelopes. Useful when notes are hanging and nothing else
        helps.
        """
        if self.closed:
            return
        for msg in panic_messages():
            self.send(msg)