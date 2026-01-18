from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.vlan_parser import vlan_parser
@pass_environment
def _vlan_parser(*args, **kwargs):
    """Extend vlan data"""
    keys = ['data', 'first_line_len', 'other_line_len']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='vlan_parser')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return vlan_parser(**updated_data)