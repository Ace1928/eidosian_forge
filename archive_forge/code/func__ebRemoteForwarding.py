import fcntl
import getpass
import os
import signal
import struct
import sys
import tty
from typing import List, Tuple
from twisted.conch.client import connect, default
from twisted.conch.client.options import ConchOptions
from twisted.conch.error import ConchError
from twisted.conch.ssh import channel, common, connection, forwarding, session
from twisted.internet import reactor, stdio, task
from twisted.python import log, usage
from twisted.python.compat import ioType, networkString
def _ebRemoteForwarding(self, f, remotePort, hostport):
    log.msg(f'remote forwarding {remotePort}:{hostport} failed')
    log.msg(f)