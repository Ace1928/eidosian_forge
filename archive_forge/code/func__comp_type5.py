from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.comp_type5 import comp_type5
@pass_environment
def _comp_type5(*args, **kwargs):
    """Extend vlan data"""
    keys = ['unencrypted_password', 'encrypted_password', 'return_original']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='comp_type5')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return comp_type5(**updated_data)