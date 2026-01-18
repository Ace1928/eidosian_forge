from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.virtual.freebsd import FreeBSDVirtual, VirtualCollector
class DragonFlyVirtualCollector(VirtualCollector):
    _fact_class = FreeBSDVirtual
    _platform = 'DragonFly'