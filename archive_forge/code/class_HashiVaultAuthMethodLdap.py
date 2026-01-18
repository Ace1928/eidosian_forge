from __future__ import absolute_import, division, print_function
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultAuthMethodBase
class HashiVaultAuthMethodLdap(HashiVaultAuthMethodBase):
    """HashiVault option group class for auth: ldap"""
    NAME = 'ldap'
    OPTIONS = ['username', 'password', 'mount_point']

    def __init__(self, option_adapter, warning_callback, deprecate_callback):
        super(HashiVaultAuthMethodLdap, self).__init__(option_adapter, warning_callback, deprecate_callback)

    def validate(self):
        self.validate_by_required_fields('username', 'password')

    def authenticate(self, client, use_token=True):
        params = self._options.get_filled_options(*self.OPTIONS)
        try:
            response = client.auth.ldap.login(use_token=use_token, **params)
        except (NotImplementedError, AttributeError):
            self.warn("HVAC should be updated to version 0.7.0 or higher. Deprecated method 'auth_ldap' will be used.")
            response = client.auth_ldap(use_token=use_token, **params)
        return response