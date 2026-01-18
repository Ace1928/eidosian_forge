import errno
import os
import socket
import ssl
import stat
import sys
import time
from gunicorn import util
def _sock_type(addr):
    if isinstance(addr, tuple):
        if util.is_ipv6(addr[0]):
            sock_type = TCP6Socket
        else:
            sock_type = TCPSocket
    elif isinstance(addr, (str, bytes)):
        sock_type = UnixSocket
    else:
        raise TypeError('Unable to create socket from: %r' % addr)
    return sock_type