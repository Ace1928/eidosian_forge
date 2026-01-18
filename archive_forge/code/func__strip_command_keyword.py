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
def _strip_command_keyword(self, keyword, arg):
    keylen = len(keyword)
    if arg[:keylen].upper() == keyword:
        return arg[keylen:].strip()
    return ''