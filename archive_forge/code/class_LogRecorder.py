import errno
import logging
import os
import platform
import socket
import ssl
import sys
import warnings
import pytest
from urllib3 import util
from urllib3.exceptions import HTTPWarning
from urllib3.packages import six
from urllib3.util import ssl_
class LogRecorder(object):

    def __init__(self, target=logging.root):
        super(LogRecorder, self).__init__()
        self._target = target
        self._handler = _ListHandler()

    @property
    def records(self):
        return self._handler.records

    def install(self):
        self._target.addHandler(self._handler)

    def uninstall(self):
        self._target.removeHandler(self._handler)

    def __enter__(self):
        self.install()
        return self.records

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()
        return False