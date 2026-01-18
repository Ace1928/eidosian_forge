from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _load_ips_ifconfig() -> None:
    """load ip addresses from `ifconfig` output (posix)"""
    try:
        out = _get_output('ifconfig')
    except OSError:
        out = _get_output('/sbin/ifconfig')
    lines = out.splitlines()
    addrs = []
    for line in lines:
        m = _ifconfig_ipv4_pat.match(line.strip())
        if m:
            addrs.append(m.group(1))
    _populate_from_list(addrs)