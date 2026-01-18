from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
class LdapSearch(LdapGeneric):

    def __init__(self, module):
        LdapGeneric.__init__(self, module)
        self.filterstr = self.module.params['filter']
        self.attrlist = []
        self.page_size = self.module.params['page_size']
        self._load_scope()
        self._load_attrs()
        self._load_schema()
        self._base64_attributes = set(self.module.params['base64_attributes'] or [])

    def _load_schema(self):
        self.schema = self.module.params['schema']
        if self.schema:
            self.attrsonly = 1
        else:
            self.attrsonly = 0

    def _load_scope(self):
        spec = dict(base=ldap.SCOPE_BASE, onelevel=ldap.SCOPE_ONELEVEL, subordinate=ldap.SCOPE_SUBORDINATE, children=ldap.SCOPE_SUBTREE)
        self.scope = spec[self.module.params['scope']]

    def _load_attrs(self):
        self.attrlist = self.module.params['attrs'] or None

    def main(self):
        results = self.perform_search()
        self.module.exit_json(changed=False, results=results)

    def perform_search(self):
        ldap_entries = []
        controls = []
        if self.page_size > 0:
            controls.append(ldap.controls.libldap.SimplePagedResultsControl(True, size=self.page_size, cookie=''))
        try:
            while True:
                response = self.connection.search_ext(self.dn, self.scope, filterstr=self.filterstr, attrlist=self.attrlist, attrsonly=self.attrsonly, serverctrls=controls)
                rtype, results, rmsgid, serverctrls = self.connection.result3(response)
                for result in results:
                    if isinstance(result[1], dict):
                        if self.schema:
                            ldap_entries.append(dict(dn=result[0], attrs=list(result[1].keys())))
                        else:
                            ldap_entries.append(_extract_entry(result[0], result[1], self._base64_attributes))
                cookies = [c.cookie for c in serverctrls if c.controlType == ldap.controls.libldap.SimplePagedResultsControl.controlType]
                if self.page_size > 0 and cookies and cookies[0]:
                    controls[0].cookie = cookies[0]
                else:
                    return ldap_entries
        except ldap.NO_SUCH_OBJECT:
            self.module.fail_json(msg='Base not found: {0}'.format(self.dn))