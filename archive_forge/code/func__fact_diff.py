from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_text
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.fact_diff import fact_diff
@pass_environment
def _fact_diff(*args, **kwargs):
    """Find the difference between currently set facts"""
    keys = ['before', 'after', 'plugin']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='fact_diff')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    res = fact_diff(**updated_data)
    return to_text(res)