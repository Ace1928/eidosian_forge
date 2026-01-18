from __future__ import absolute_import, division, print_function
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultAuthMethodBase
class HashiVaultAuthMethodUserpass(HashiVaultAuthMethodBase):
    """HashiVault option group class for auth: userpass"""
    NAME = 'userpass'
    OPTIONS = ['username', 'password', 'mount_point']

    def __init__(self, option_adapter, warning_callback, deprecate_callback):
        super(HashiVaultAuthMethodUserpass, self).__init__(option_adapter, warning_callback, deprecate_callback)

    def validate(self):
        self.validate_by_required_fields('username', 'password')

    def authenticate(self, client, use_token=True):
        params = self._options.get_filled_options(*self.OPTIONS)
        try:
            response = client.auth.userpass.login(**params)
        except (NotImplementedError, AttributeError):
            self.warn("HVAC should be updated to version 0.9.6 or higher. Deprecated method 'auth_userpass' will be used.")
            response = client.auth_userpass(**params)
        if use_token:
            client.token = response['auth']['client_token']
        return response