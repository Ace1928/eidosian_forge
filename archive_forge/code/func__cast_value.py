from __future__ import absolute_import, division, print_function
import re
import shlex
import time
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six import string_types, text_type
from ansible.module_utils.six.moves import zip
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def _cast_value(self, value):
    if value in BOOLEANS_TRUE:
        return True
    elif value in BOOLEANS_FALSE:
        return False
    elif re.match('^\\d+\\.d+$', value):
        return float(value)
    elif re.match('^\\d+$', value):
        return int(value)
    else:
        return text_type(value)