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
class FileStrategy(BaseStrategy):
    FILE = '/etc/hostname'

    def get_permanent_hostname(self):
        if not os.path.isfile(self.FILE):
            return ''
        try:
            return get_file_content(self.FILE, default='', strip=True)
        except Exception as e:
            self.module.fail_json(msg='failed to read hostname: %s' % to_native(e), exception=traceback.format_exc())

    def set_permanent_hostname(self, name):
        try:
            with open(self.FILE, 'w+') as f:
                f.write('%s\n' % name)
        except Exception as e:
            self.module.fail_json(msg='failed to update hostname: %s' % to_native(e), exception=traceback.format_exc())