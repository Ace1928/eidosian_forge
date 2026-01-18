import os
import shutil
import tempfile
from twisted.internet.protocol import Protocol
from twisted.internet.test.test_serialport import DoNothing
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.trial import unittest
class RegularFileSerial(serial.Serial):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_args = args
        self.captured_kwargs = kwargs

    def _reconfigurePort(self):
        pass

    def _reconfigure_port(self):
        pass