from __future__ import absolute_import, division, print_function
import binascii
import contextlib
import datetime
import errno
import math
import mmap
import os
import re
import select
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.compat.datetime import utcnow
def _convert_host_to_ip(host):
    """
    Perform forward DNS resolution on host, IP will give the same IP

    Args:
        host: String with either hostname, IPv4, or IPv6 address

    Returns:
        List of tuples containing address family and IP
    """
    addrinfo = socket.getaddrinfo(host, 80, 0, 0, socket.SOL_TCP)
    ips = []
    for family, socktype, proto, canonname, sockaddr in addrinfo:
        ip = sockaddr[0]
        ips.append((family, ip))
        if family == socket.AF_INET:
            ips.append((socket.AF_INET6, '::ffff:' + ip))
    return ips