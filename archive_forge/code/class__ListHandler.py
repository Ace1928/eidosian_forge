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
class _ListHandler(logging.Handler):

    def __init__(self):
        super(_ListHandler, self).__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)