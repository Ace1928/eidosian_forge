from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.hash_salt import hash_salt
@pass_environment
def _hash_salt(*args, **kwargs):
    """Extend vlan data"""
    keys = ['password']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='hash_salt')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return hash_salt(**updated_data)