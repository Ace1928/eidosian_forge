from __future__ import unicode_literals
from .utils import DummyContext, is_windows
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import io
import os
import sys
class PipeInput(Input):
    """
    Input that is send through a pipe.
    This is useful if we want to send the input programatically into the
    interface, but still use the eventloop.

    Usage::

        input = PipeInput()
        input.send('inputdata')
    """

    def __init__(self):
        self._r, self._w = os.pipe()

    def fileno(self):
        return self._r

    def read(self):
        return os.read(self._r)

    def send_text(self, data):
        """ Send text to the input. """
        os.write(self._w, data.encode('utf-8'))
    send = send_text

    def raw_mode(self):
        return DummyContext()

    def cooked_mode(self):
        return DummyContext()

    def close(self):
        """ Close pipe fds. """
        os.close(self._r)
        os.close(self._w)
        self._r = None
        self._w = None