from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def __concat_attribute_sequence(trace_path):
    rdata = ''
    if type(trace_path) is not list:
        raise AssertionError()
    if len(trace_path) >= 1:
        rdata += str(trace_path[0])
    for item in trace_path[1:]:
        rdata += '.' + str(item)
    return rdata