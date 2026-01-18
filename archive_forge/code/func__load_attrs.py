from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
def _load_attrs(self):
    """ Turn attribute's value to array. """
    attrs = {}
    for name, value in self.module.params['attributes'].items():
        if isinstance(value, list):
            attrs[name] = list(map(to_bytes, value))
        else:
            attrs[name] = [to_bytes(value)]
    return attrs