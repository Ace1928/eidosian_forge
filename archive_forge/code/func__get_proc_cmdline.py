from __future__ import (absolute_import, division, print_function)
import shlex
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector
def _get_proc_cmdline(self):
    return get_file_content('/proc/cmdline')