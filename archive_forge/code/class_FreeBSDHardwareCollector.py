from __future__ import (absolute_import, division, print_function)
import os
import json
import re
import struct
import time
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.timeout import TimeoutError, timeout
from ansible.module_utils.facts.utils import get_file_content, get_mount_size
class FreeBSDHardwareCollector(HardwareCollector):
    _fact_class = FreeBSDHardware
    _platform = 'FreeBSD'