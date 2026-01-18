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
def _lsblk_uuid(self):
    uuids = {}
    lsblk_path = self.module.get_bin_path('lsblk')
    if not lsblk_path:
        return uuids
    rc, out, err = self._run_lsblk(lsblk_path)
    if rc != 0:
        return uuids
    for lsblk_line in out.splitlines():
        if not lsblk_line:
            continue
        line = lsblk_line.strip()
        fields = line.rsplit(None, 1)
        if len(fields) < 2:
            continue
        device_name, uuid = (fields[0].strip(), fields[1].strip())
        if device_name in uuids:
            continue
        uuids[device_name] = uuid
    return uuids