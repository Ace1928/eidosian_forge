from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.facts.virtual.base import Virtual, VirtualCollector
class SunOSVirtualCollector(VirtualCollector):
    _fact_class = SunOSVirtual
    _platform = 'SunOS'