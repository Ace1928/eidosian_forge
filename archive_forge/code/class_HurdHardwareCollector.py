from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.timeout import TimeoutError
from ansible.module_utils.facts.hardware.base import HardwareCollector
from ansible.module_utils.facts.hardware.linux import LinuxHardware
class HurdHardwareCollector(HardwareCollector):
    _fact_class = HurdHardware
    _platform = 'GNU'