from __future__ import absolute_import, division, print_function
import types
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
@pass_environment
def _ipwrap(*args, **kwargs):
    """This filter is designed to Wrap IPv6 addresses in [ ] brackets."""
    keys = ['value']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    try:
        if isinstance(data['value'], str):
            pass
        elif isinstance(data['value'], list):
            pass
        elif isinstance(data['value'], bool):
            pass
        else:
            raise AnsibleFilterError('Unrecognized type <{0}> for ipwrap filter <{1}>'.format(type(data['value']), 'value'))
    except (TypeError, ValueError):
        raise AnsibleFilterError('Unrecognized type <{0}> for ipwrap filter <{1}>'.format(type(data['value']), 'value'))
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='ipwrap')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return ipwrap(**updated_data)