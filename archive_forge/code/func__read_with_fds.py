import array
from collections import deque
from errno import ECONNRESET
import functools
from itertools import count
import os
from selectors import DefaultSelector, EVENT_READ
import socket
import time
from typing import Optional
from warnings import warn
from jeepney import Parser, Message, MessageType, HeaderFields
from jeepney.auth import Authenticator, BEGIN
from jeepney.bus import get_bus
from jeepney.fds import FileDescriptor, fds_buf_size
from jeepney.wrappers import ProxyBase, unwrap_msg
from jeepney.routing import Router
from jeepney.bus_messages import message_bus
from .common import MessageFilters, FilterHandle, check_replyable
def _read_with_fds(self):
    nbytes = self.parser.bytes_desired()
    data, ancdata, flags, _ = self.sock.recvmsg(nbytes, fds_buf_size())
    if flags & getattr(socket, 'MSG_CTRUNC', 0):
        self.close()
        raise RuntimeError('Unable to receive all file descriptors')
    return (unwrap_read(data), FileDescriptor.from_ancdata(ancdata))