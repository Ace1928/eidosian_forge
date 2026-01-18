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
def _method_call(self, make_msg):

    @functools.wraps(make_msg)
    def inner(*args, **kwargs):
        timeout = kwargs.pop('_timeout', self._timeout)
        msg = make_msg(*args, **kwargs)
        assert msg.header.message_type is MessageType.method_call
        return unwrap_msg(self._connection.send_and_get_reply(msg, timeout=timeout))
    return inner