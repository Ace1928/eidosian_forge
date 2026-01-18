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
def _set_post_data_state(self):
    """Reset state variables to their post-DATA state."""
    self.smtp_state = self.COMMAND
    self.mailfrom = None
    self.rcpttos = []
    self.require_SMTPUTF8 = False
    self.num_bytes = 0
    self.set_terminator(b'\r\n')