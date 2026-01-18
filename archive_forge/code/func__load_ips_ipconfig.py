from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _load_ips_ipconfig() -> None:
    """load ip addresses from `ipconfig` output (Windows)"""
    out = _get_output('ipconfig')
    lines = out.splitlines()
    addrs = []
    for line in lines:
        m = _ipconfig_ipv4_pat.match(line.strip())
        if m:
            addrs.append(m.group(1))
    _populate_from_list(addrs)