from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
def _is_entry_present(self):
    try:
        self.connection.search_s(self.dn, ldap.SCOPE_BASE)
    except ldap.NO_SUCH_OBJECT:
        is_present = False
    else:
        is_present = True
    return is_present