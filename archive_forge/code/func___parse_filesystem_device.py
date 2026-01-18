from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __parse_filesystem_device(self, line):
    return re.sub('^.*path\\s', '', line)