from __future__ import (absolute_import, division, print_function)
import datetime
from ansible.module_utils.six import string_types, integer_types
from ansible.module_utils._text import to_native
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
def _fix_sort_parameter(self, sort_parameter):
    if sort_parameter is None:
        return sort_parameter
    if not isinstance(sort_parameter, list):
        raise AnsibleError(u'Error. Sort parameters must be a list, not [ {0} ]'.format(sort_parameter))
    for item in sort_parameter:
        self._convert_sort_string_to_constant(item)
    return sort_parameter