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
def _get_method_object(self, method=None):
    if method is None:
        method = self._options.get_option('auth_method')
    try:
        o_method = self._selector[method]
    except KeyError:
        raise NotImplementedError("auth method '%s' is not implemented in HashiVaultAuthenticator" % method)
    return o_method