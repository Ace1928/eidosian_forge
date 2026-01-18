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
def _print_message_content(self, peer, data):
    inheaders = 1
    lines = data.splitlines()
    for line in lines:
        if inheaders and (not line):
            peerheader = 'X-Peer: ' + peer[0]
            if not isinstance(data, str):
                peerheader = repr(peerheader.encode('utf-8'))
            print(peerheader)
            inheaders = 0
        if not isinstance(data, str):
            line = repr(line)
        print(line)