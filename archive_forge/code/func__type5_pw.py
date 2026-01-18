from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.type5_pw import type5_pw
@pass_environment
def _type5_pw(*args, **kwargs):
    """Extend vlan data"""
    keys = ['password', 'salt']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='type5_pw')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return type5_pw(**updated_data)