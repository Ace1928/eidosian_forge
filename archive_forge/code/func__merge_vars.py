from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def _merge_vars(self, search_pattern, initial_value, variables):
    display.vvv('Merge variables with {0}: {1}'.format(self._pattern_type, search_pattern))
    var_merge_names = sorted([key for key in variables.keys() if self._var_matches(key, search_pattern)])
    display.vvv('The following variables will be merged: {0}'.format(var_merge_names))
    prev_var_type = None
    result = None
    if initial_value is not None:
        prev_var_type = _verify_and_get_type(initial_value)
        result = initial_value
    for var_name in var_merge_names:
        var_value = self._templar.template(variables[var_name])
        var_type = _verify_and_get_type(var_value)
        if prev_var_type is None:
            prev_var_type = var_type
        elif prev_var_type != var_type:
            raise AnsibleError('Unable to merge, not all variables are of the same type')
        if result is None:
            result = var_value
            continue
        if var_type == 'dict':
            result = self._merge_dict(var_value, result, [var_name])
        else:
            result += var_value
    return result