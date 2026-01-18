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
def _read_some_data(self, timeout=None):
    for key, ev in self.selector.select(timeout):
        if key == self.select_key:
            if self.enable_fds:
                return self._read_with_fds()
            else:
                return (unwrap_read(self.sock.recv(4096)), [])
    raise TimeoutError