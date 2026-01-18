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
def _scrub_hostname(self, name):
    """
        LocalHostName only accepts valid DNS characters while HostName and ComputerName
        accept a much wider range of characters. This function aims to mimic how macOS
        translates a friendly name to the LocalHostName.
        """
    name = to_text(name)
    replace_chars = u'\'"~`!@#$%^&*(){}[]/=?+\\|-_ '
    delete_chars = u".'"
    table = self._make_translation(replace_chars, u'-' * len(replace_chars), delete_chars)
    name = name.translate(table)
    while '-' * 2 in name:
        name = name.replace('-' * 2, '')
    name = name.rstrip('-')
    return name