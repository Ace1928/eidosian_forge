import socket
import sys
import threading
from debugpy.common import log
from debugpy.common.util import hide_thread_from_debugger
def _new_sock():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except (AttributeError, OSError):
        pass
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
    except (AttributeError, OSError):
        pass
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)
    except (AttributeError, OSError):
        pass
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
    except (AttributeError, OSError):
        pass
    return sock