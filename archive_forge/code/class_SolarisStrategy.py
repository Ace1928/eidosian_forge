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
class SolarisStrategy(BaseStrategy):
    """
    This is a Solaris11 or later Hostname manipulation strategy class - it
    execute hostname command.
    """
    COMMAND = 'hostname'

    def __init__(self, module):
        super(SolarisStrategy, self).__init__(module)
        self.hostname_cmd = self.module.get_bin_path(self.COMMAND, True)

    def set_current_hostname(self, name):
        cmd_option = '-t'
        cmd = [self.hostname_cmd, cmd_option, name]
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))

    def get_permanent_hostname(self):
        fmri = 'svc:/system/identity:node'
        pattern = 'config/nodename'
        cmd = '/usr/sbin/svccfg -s %s listprop -o value %s' % (fmri, pattern)
        rc, out, err = self.module.run_command(cmd, use_unsafe_shell=True)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))
        return to_native(out).strip()

    def set_permanent_hostname(self, name):
        cmd = [self.hostname_cmd, name]
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))