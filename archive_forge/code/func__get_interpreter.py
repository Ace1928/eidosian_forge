from __future__ import absolute_import, division, print_function
import errno
import json
import shlex
import shutil
import os
import subprocess
import sys
import traceback
import signal
import time
import syslog
import multiprocessing
from ansible.module_utils.common.text.converters import to_text, to_bytes
def _get_interpreter(module_path):
    with open(module_path, 'rb') as module_fd:
        head = module_fd.read(1024)
        if head[0:2] != b'#!':
            return None
        return head[2:head.index(b'\n')].strip().split(b' ')