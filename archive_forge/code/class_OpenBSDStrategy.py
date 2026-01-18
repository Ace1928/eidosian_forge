from __future__ import absolute_import, division, print_function
import os
import platform
import socket
import traceback
import ansible.module_utils.compat.typing as t
from ansible.module_utils.basic import (
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.facts.system.service_mgr import ServiceMgrFactCollector
from ansible.module_utils.facts.utils import get_file_lines, get_file_content
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import PY3, text_type
class OpenBSDStrategy(FileStrategy):
    """
    This is a OpenBSD family Hostname manipulation strategy class - it edits
    the /etc/myname file for the permanent hostname and executes hostname
    command for the current hostname.
    """
    FILE = '/etc/myname'
    COMMAND = 'hostname'

    def __init__(self, module):
        super(OpenBSDStrategy, self).__init__(module)
        self.hostname_cmd = self.module.get_bin_path(self.COMMAND, True)

    def get_current_hostname(self):
        cmd = [self.hostname_cmd]
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))
        return to_native(out).strip()

    def set_current_hostname(self, name):
        cmd = [self.hostname_cmd, name]
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))