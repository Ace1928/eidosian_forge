from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes, to_text
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
import re
def _normalize_values(self, values):
    """ Normalize attribute's values. """
    norm_values = []
    if isinstance(values, list):
        if self.ordered:
            norm_values = list(map(to_bytes, self._order_values(list(map(str, values)))))
        else:
            norm_values = list(map(to_bytes, values))
    else:
        norm_values = [to_bytes(str(values))]
    return norm_values