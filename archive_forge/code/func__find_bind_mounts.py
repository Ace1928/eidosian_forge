from __future__ import (absolute_import, division, print_function)
import collections
import errno
import glob
import json
import os
import re
import sys
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.utils import get_file_content, get_file_lines, get_mount_size
from ansible.module_utils.six import iteritems
from ansible.module_utils.facts import timeout
def _find_bind_mounts(self):
    bind_mounts = set()
    findmnt_path = self.module.get_bin_path('findmnt')
    if not findmnt_path:
        return bind_mounts
    rc, out, err = self._run_findmnt(findmnt_path)
    if rc != 0:
        return bind_mounts
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        if self.BIND_MOUNT_RE.match(fields[1]):
            bind_mounts.add(fields[0])
    return bind_mounts