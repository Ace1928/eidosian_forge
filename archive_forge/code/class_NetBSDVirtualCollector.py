from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.facts.virtual.base import Virtual, VirtualCollector
from ansible.module_utils.facts.virtual.sysctl import VirtualSysctlDetectionMixin
class NetBSDVirtualCollector(VirtualCollector):
    _fact_class = NetBSDVirtual
    _platform = 'NetBSD'