from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
def _connect_to_ldap(self):
    if not self.verify_cert:
        ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_NEVER)
    if self.ca_path:
        ldap.set_option(ldap.OPT_X_TLS_CACERTFILE, self.ca_path)
    if self.client_cert and self.client_key:
        ldap.set_option(ldap.OPT_X_TLS_CERTFILE, self.client_cert)
        ldap.set_option(ldap.OPT_X_TLS_KEYFILE, self.client_key)
    connection = ldap.initialize(self.server_uri)
    if self.referrals_chasing == 'disabled':
        connection.set_option(ldap.OPT_REFERRALS, 0)
    if self.start_tls:
        try:
            connection.start_tls_s()
        except ldap.LDAPError as e:
            self.fail('Cannot start TLS.', e)
    try:
        if self.bind_dn is not None:
            connection.simple_bind_s(self.bind_dn, self.bind_pw)
        else:
            klass = SASCL_CLASS.get(self.sasl_class, ldap.sasl.external)
            connection.sasl_interactive_bind_s('', klass())
    except ldap.LDAPError as e:
        self.fail('Cannot bind to the server.', e)
    return connection