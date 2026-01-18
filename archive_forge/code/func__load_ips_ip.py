from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _load_ips_ip() -> None:
    """load ip addresses from `ip addr` output (Linux)"""
    out = _get_output(['ip', '-f', 'inet', 'addr'])
    lines = out.splitlines()
    addrs = []
    for line in lines:
        blocks = line.lower().split()
        if len(blocks) >= 2 and blocks[0] == 'inet':
            addrs.append(blocks[1].split('/')[0])
    _populate_from_list(addrs)