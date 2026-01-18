from __future__ import (absolute_import, division, print_function)
import os
import os.path
import subprocess
import traceback
from ansible.errors import AnsibleError
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
@staticmethod
def _search_executable(executable):
    try:
        return get_bin_path(executable)
    except ValueError:
        raise AnsibleError('%s command not found in PATH' % executable)