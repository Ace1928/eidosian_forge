from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __parse_subvolume_list_record(self, item):
    return {'id': int(item[0]), 'parent': int(item[2]), 'path': normalize_subvolume_path(item[5])}