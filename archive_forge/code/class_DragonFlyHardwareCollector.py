from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.hardware.base import HardwareCollector
from ansible.module_utils.facts.hardware.freebsd import FreeBSDHardware
class DragonFlyHardwareCollector(HardwareCollector):
    _fact_class = FreeBSDHardware
    _platform = 'DragonFly'