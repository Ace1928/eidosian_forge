from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import (
class HashiVaultAuthMethodToken(HashiVaultAuthMethodBase):
    """HashiVault option group class for auth: userpass"""
    NAME = 'token'
    OPTIONS = ['token', 'token_path', 'token_file', 'token_validate']
    _LATE_BINDING_ENV_VAR_OPTIONS = {'token': dict(env=['VAULT_TOKEN']), 'token_path': dict(env=['HOME'])}

    def __init__(self, option_adapter, warning_callback, deprecate_callback):
        super(HashiVaultAuthMethodToken, self).__init__(option_adapter, warning_callback, deprecate_callback)

    def _simulate_login_response(self, token, lookup_response=None):
        """returns a similar structure to a login method's return, optionally incorporating a lookup-self response"""
        response = {'auth': {'client_token': token}}
        if lookup_response is None:
            return response
        response.update(lookup_response, auth=response['auth'])
        response['auth'].update(lookup_response['data'])
        metadata = response['auth'].pop('meta', None)
        if metadata:
            response['auth']['metadata'] = metadata
        return response

    def validate(self):
        self.process_late_binding_env_vars(self._LATE_BINDING_ENV_VAR_OPTIONS)
        if self._options.get_option_default('token') is None and self._options.get_option_default('token_path') is not None:
            token_filename = os.path.join(self._options.get_option('token_path'), self._options.get_option('token_file'))
            if os.path.exists(token_filename):
                if not os.path.isfile(token_filename):
                    raise HashiVaultValueError("The Vault token file '%s' was found but is not a file." % token_filename)
                with open(token_filename) as token_file:
                    self._options.set_option('token', token_file.read().strip())
        if self._options.get_option_default('token') is None:
            raise HashiVaultValueError('No Vault Token specified or discovered.')

    def authenticate(self, client, use_token=True, lookup_self=False):
        token = self._options.get_option('token')
        validate = self._options.get_option_default('token_validate')
        response = None
        if use_token:
            client.token = token
            if lookup_self or validate:
                from hvac import exceptions
                try:
                    try:
                        response = client.auth.token.lookup_self()
                    except (NotImplementedError, AttributeError):
                        response = client.lookup_token()
                except (exceptions.Forbidden, exceptions.InvalidPath, exceptions.InvalidRequest):
                    if validate:
                        raise HashiVaultValueError('Invalid Vault Token Specified.')
        return self._simulate_login_response(token, response)