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
class SLESHostname(Hostname):
    platform = 'Linux'
    distribution = 'Sles'
    try:
        distribution_version = get_distribution_version()
        if distribution_version and 10 <= float(distribution_version) <= 12:
            strategy_class = SLESStrategy
        else:
            raise ValueError()
    except ValueError:
        strategy_class = UnimplementedStrategy