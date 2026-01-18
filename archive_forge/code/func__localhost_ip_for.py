from __future__ import annotations
import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
import attrs
import trio
from trio._util import NoPublicConstructor, final
def _localhost_ip_for(family: int) -> IPAddress:
    if family == trio.socket.AF_INET:
        return ipaddress.ip_address('127.0.0.1')
    elif family == trio.socket.AF_INET6:
        return ipaddress.ip_address('::1')
    raise NotImplementedError('Unhandled ip address family')