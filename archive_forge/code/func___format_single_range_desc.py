from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def __format_single_range_desc(one_range):
    if len(one_range) != 2:
        raise BaseException('Incorrect version range, expecting [start, end]: ' + str(one_range))
    if one_range[0] == one_range[1]:
        return one_range[0]
    elif one_range[1] == '':
        return one_range[0] + ' -> latest'
    else:
        return one_range[0] + ' -> ' + one_range[1]