from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
class LdapPasswd(LdapGeneric):

    def __init__(self, module):
        LdapGeneric.__init__(self, module)
        self.passwd = self.module.params['passwd']

    def passwd_check(self):
        try:
            tmp_con = ldap.initialize(self.server_uri)
        except ldap.LDAPError as e:
            self.fail('Cannot initialize LDAP connection', e)
        if self.start_tls:
            try:
                tmp_con.start_tls_s()
            except ldap.LDAPError as e:
                self.fail('Cannot start TLS.', e)
        try:
            tmp_con.simple_bind_s(self.dn, self.passwd)
        except ldap.INVALID_CREDENTIALS:
            return True
        except ldap.LDAPError as e:
            self.fail('Cannot bind to the server.', e)
        else:
            return False
        finally:
            tmp_con.unbind()

    def passwd_set(self):
        if not self.passwd_check():
            return False
        try:
            self.connection.passwd_s(self.dn, None, self.passwd)
        except ldap.LDAPError as e:
            self.fail('Unable to set password', e)
        return True