from __future__ import absolute_import, division, print_function
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_approle import HashiVaultAuthMethodApprole
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_aws_iam import HashiVaultAuthMethodAwsIam
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_azure import HashiVaultAuthMethodAzure
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_cert import HashiVaultAuthMethodCert
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_jwt import HashiVaultAuthMethodJwt
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_ldap import HashiVaultAuthMethodLdap
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_none import HashiVaultAuthMethodNone
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_token import HashiVaultAuthMethodToken
from ansible_collections.community.hashi_vault.plugins.module_utils._auth_method_userpass import HashiVaultAuthMethodUserpass
class HashiVaultAuthenticator:
    ARGSPEC = dict(auth_method=dict(type='str', default='token', choices=['token', 'userpass', 'ldap', 'approle', 'aws_iam', 'azure', 'jwt', 'cert', 'none']), mount_point=dict(type='str'), token=dict(type='str', no_log=True, default=None), token_path=dict(type='str', default=None, no_log=False), token_file=dict(type='str', default='.vault-token'), token_validate=dict(type='bool', default=False), username=dict(type='str'), password=dict(type='str', no_log=True), role_id=dict(type='str'), secret_id=dict(type='str', no_log=True), jwt=dict(type='str', no_log=True), aws_profile=dict(type='str', aliases=['boto_profile']), aws_access_key=dict(type='str', aliases=['aws_access_key_id'], no_log=False), aws_secret_key=dict(type='str', aliases=['aws_secret_access_key'], no_log=True), aws_security_token=dict(type='str', no_log=False), region=dict(type='str'), aws_iam_server_id=dict(type='str'), azure_tenant_id=dict(type='str'), azure_client_id=dict(type='str'), azure_client_secret=dict(type='str', no_log=True), azure_resource=dict(type='str', default='https://management.azure.com/'), cert_auth_private_key=dict(type='path', no_log=False), cert_auth_public_key=dict(type='path'))

    def __init__(self, option_adapter, warning_callback, deprecate_callback):
        self._options = option_adapter
        self._selector = {'approle': HashiVaultAuthMethodApprole(option_adapter, warning_callback, deprecate_callback), 'aws_iam': HashiVaultAuthMethodAwsIam(option_adapter, warning_callback, deprecate_callback), 'azure': HashiVaultAuthMethodAzure(option_adapter, warning_callback, deprecate_callback), 'cert': HashiVaultAuthMethodCert(option_adapter, warning_callback, deprecate_callback), 'jwt': HashiVaultAuthMethodJwt(option_adapter, warning_callback, deprecate_callback), 'ldap': HashiVaultAuthMethodLdap(option_adapter, warning_callback, deprecate_callback), 'none': HashiVaultAuthMethodNone(option_adapter, warning_callback, deprecate_callback), 'token': HashiVaultAuthMethodToken(option_adapter, warning_callback, deprecate_callback), 'userpass': HashiVaultAuthMethodUserpass(option_adapter, warning_callback, deprecate_callback)}
        self.warn = warning_callback
        self.deprecate = deprecate_callback

    def _get_method_object(self, method=None):
        if method is None:
            method = self._options.get_option('auth_method')
        try:
            o_method = self._selector[method]
        except KeyError:
            raise NotImplementedError("auth method '%s' is not implemented in HashiVaultAuthenticator" % method)
        return o_method

    def validate(self, *args, **kwargs):
        method = self._get_method_object(kwargs.pop('method', None))
        method.validate(*args, **kwargs)

    def authenticate(self, *args, **kwargs):
        method = self._get_method_object(kwargs.pop('method', None))
        return method.authenticate(*args, **kwargs)