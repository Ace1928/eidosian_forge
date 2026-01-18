from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
def _load_scope(self):
    spec = dict(base=ldap.SCOPE_BASE, onelevel=ldap.SCOPE_ONELEVEL, subordinate=ldap.SCOPE_SUBORDINATE, children=ldap.SCOPE_SUBTREE)
    self.scope = spec[self.module.params['scope']]