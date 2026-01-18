import io
import os
import re
import sys
from ._core import Process
def _get_ppid(pid, name):
    path = os.path.join('/proc', str(pid), name)
    with io.open(path, encoding='ascii', errors='replace') as f:
        parts = STAT_PATTERN.findall(f.read())
    if _use_bsd_stat_format():
        return parts[BSD_STAT_PPID]
    return parts[LINUX_STAT_PPID]