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
def _enterRawMode():
    global _inRawMode, _savedRawMode
    if _inRawMode:
        return
    fd = sys.stdin.fileno()
    try:
        old = tty.tcgetattr(fd)
        new = old[:]
    except BaseException:
        log.msg('not a typewriter!')
    else:
        new[0] = new[0] | tty.IGNPAR
        new[0] = new[0] & ~(tty.ISTRIP | tty.INLCR | tty.IGNCR | tty.ICRNL | tty.IXON | tty.IXANY | tty.IXOFF)
        if hasattr(tty, 'IUCLC'):
            new[0] = new[0] & ~tty.IUCLC
        new[3] = new[3] & ~(tty.ISIG | tty.ICANON | tty.ECHO | tty.ECHO | tty.ECHOE | tty.ECHOK | tty.ECHONL)
        if hasattr(tty, 'IEXTEN'):
            new[3] = new[3] & ~tty.IEXTEN
        new[1] = new[1] & ~tty.OPOST
        new[6][tty.VMIN] = 1
        new[6][tty.VTIME] = 0
        _savedRawMode = old
        tty.tcsetattr(fd, tty.TCSANOW, new)
        _inRawMode = 1