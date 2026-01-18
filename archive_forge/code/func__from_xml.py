from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.from_xml import from_xml
@pass_environment
def _from_xml(*args, **kwargs):
    """Convert the given data from xml to json."""
    keys = ['data', 'engine']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='from_xml')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return from_xml(**updated_data)