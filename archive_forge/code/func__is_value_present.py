from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes, to_text
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
import re
def _is_value_present(self, name, value):
    """ True if the target attribute has the given value. """
    try:
        escaped_value = ldap.filter.escape_filter_chars(to_text(value))
        filterstr = '(%s=%s)' % (name, escaped_value)
        dns = self.connection.search_s(self.dn, ldap.SCOPE_BASE, filterstr)
        is_present = len(dns) == 1
    except ldap.NO_SUCH_OBJECT:
        is_present = False
    return is_present