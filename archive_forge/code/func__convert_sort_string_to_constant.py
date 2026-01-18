from __future__ import (absolute_import, division, print_function)
import datetime
from ansible.module_utils.six import string_types, integer_types
from ansible.module_utils._text import to_native
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
def _convert_sort_string_to_constant(self, item):
    original_sort_order = item[1]
    sort_order = original_sort_order.upper()
    if sort_order == u'ASCENDING':
        item[1] = ASCENDING
    elif sort_order == u'DESCENDING':
        item[1] = DESCENDING