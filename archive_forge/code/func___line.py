import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
@__line.setter
def __line(self, value):
    warn("Setting __line attribute on SMTPChannel is deprecated, set 'received_lines' instead", DeprecationWarning, 2)
    self.received_lines = value