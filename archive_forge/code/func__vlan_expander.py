from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.vlan_expander import vlan_expander
@pass_environment
def _vlan_expander(*args, **kwargs):
    """Extend vlan data"""
    keys = ['data']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='vlan_expander')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return vlan_expander(**updated_data)