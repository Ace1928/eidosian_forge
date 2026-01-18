from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def _verify_and_get_type(variable):
    if isinstance(variable, list):
        return 'list'
    elif isinstance(variable, dict):
        return 'dict'
    else:
        raise AnsibleError('Not supported type detected, variable must be a list or a dict')