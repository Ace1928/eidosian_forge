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
@__addr.setter
def __addr(self, value):
    warn("Setting __addr attribute on SMTPChannel is deprecated, set 'addr' instead", DeprecationWarning, 2)
    self.addr = value