from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
@_only_once
def _load_ips(suppress_exceptions: bool=True) -> None:
    """load the IPs that point to this machine

    This function will only ever be called once.

    It will use netifaces to do it quickly if available.
    Then it will fallback on parsing the output of ifconfig / ip addr / ipconfig, as appropriate.
    Finally, it will fallback on socket.gethostbyname_ex, which can be slow.
    """
    try:
        try:
            return _load_ips_netifaces()
        except ImportError:
            pass
        if os.name == 'nt':
            try:
                return _load_ips_ipconfig()
            except (OSError, NoIPAddresses):
                pass
        else:
            try:
                return _load_ips_ip()
            except (OSError, NoIPAddresses):
                pass
            try:
                return _load_ips_ifconfig()
            except (OSError, NoIPAddresses):
                pass
        return _load_ips_gethostbyname()
    except Exception as e:
        if not suppress_exceptions:
            raise
        warn('Unexpected error discovering local network interfaces: %s' % e, stacklevel=2)
    _load_ips_dumb()