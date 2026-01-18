from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def _trace_to_string(self, trace):
    trace_string = ''
    for _trace in trace:
        trace_string += '%s%s' % (_trace, '.' if _trace != trace[-1] else '')
    return trace_string