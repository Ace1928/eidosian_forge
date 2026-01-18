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
@__fqdn.setter
def __fqdn(self, value):
    warn("Setting __fqdn attribute on SMTPChannel is deprecated, set 'fqdn' instead", DeprecationWarning, 2)
    self.fqdn = value