from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.facts.virtual.base import Virtual, VirtualCollector
from ansible.module_utils.facts.virtual.sysctl import VirtualSysctlDetectionMixin
class FreeBSDVirtualCollector(VirtualCollector):
    _fact_class = FreeBSDVirtual
    _platform = 'FreeBSD'