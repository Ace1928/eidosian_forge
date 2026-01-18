from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __filter_mountpoints_for_devices(self, mountpoints, devices):
    return [m for m in mountpoints if m['device'] in devices]