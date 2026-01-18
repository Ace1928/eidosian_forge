from __future__ import (absolute_import, division, print_function)
import re
import time
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts import timeout
from ansible.module_utils.facts.utils import get_file_content, get_mount_size
from ansible.module_utils.facts.sysctl import get_sysctl
class OpenBSDHardwareCollector(HardwareCollector):
    _fact_class = OpenBSDHardware
    _platform = 'OpenBSD'