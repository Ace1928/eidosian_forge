from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.facts.virtual.base import Virtual, VirtualCollector
from ansible.module_utils.facts.virtual.sysctl import VirtualSysctlDetectionMixin
from ansible.module_utils.facts.utils import get_file_content
class OpenBSDVirtualCollector(VirtualCollector):
    _fact_class = OpenBSDVirtual
    _platform = 'OpenBSD'