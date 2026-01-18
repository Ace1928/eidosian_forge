from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.replace_keys import replace_keys
@pass_environment
def _replace_keys(*args, **kwargs):
    """replaces specific keys with their after value from a data recursively"""
    keys = ['data', 'target', 'matching_parameter']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='replace_keys')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return replace_keys(**updated_data)