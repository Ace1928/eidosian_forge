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
class FreeBSDStrategy(BaseStrategy):
    """
    This is a FreeBSD hostname manipulation strategy class - it edits
    the /etc/rc.conf.d/hostname file.
    """
    FILE = '/etc/rc.conf.d/hostname'
    COMMAND = 'hostname'

    def __init__(self, module):
        super(FreeBSDStrategy, self).__init__(module)
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

    def get_permanent_hostname(self):
        if not os.path.isfile(self.FILE):
            return ''
        try:
            for line in get_file_lines(self.FILE):
                line = line.strip()
                if line.startswith('hostname='):
                    return line[10:].strip('"')
        except Exception as e:
            self.module.fail_json(msg='failed to read hostname: %s' % to_native(e), exception=traceback.format_exc())

    def set_permanent_hostname(self, name):
        try:
            if os.path.isfile(self.FILE):
                lines = [x.strip() for x in get_file_lines(self.FILE)]
                for i, line in enumerate(lines):
                    if line.startswith('hostname='):
                        lines[i] = 'hostname="%s"' % name
                        break
            else:
                lines = ['hostname="%s"' % name]
            with open(self.FILE, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as e:
            self.module.fail_json(msg='failed to update hostname: %s' % to_native(e), exception=traceback.format_exc())