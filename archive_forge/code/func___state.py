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
@__state.setter
def __state(self, value):
    warn("Setting __state attribute on SMTPChannel is deprecated, set 'smtp_state' instead", DeprecationWarning, 2)
    self.smtp_state = value