from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _populate_from_list(addrs: Sequence[str] | None) -> None:
    """populate local and public IPs from flat list of all IPs"""
    if not addrs:
        raise NoIPAddresses
    global LOCALHOST
    public_ips = []
    local_ips = []
    for ip in addrs:
        local_ips.append(ip)
        if not ip.startswith('127.'):
            public_ips.append(ip)
        elif not LOCALHOST:
            LOCALHOST = ip
    if not LOCALHOST or LOCALHOST == '127.0.0.1':
        LOCALHOST = '127.0.0.1'
        local_ips.insert(0, LOCALHOST)
    local_ips.extend(['0.0.0.0', ''])
    LOCAL_IPS[:] = _uniq_stable(local_ips)
    PUBLIC_IPS[:] = _uniq_stable(public_ips)