from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
class _SectionAuth(_Section):
    """Contains the properties for the 'auth' section."""
    DEFAULT_AUTH_HOST = 'https://accounts.google.com/o/oauth2/auth'
    DEFAULT_TOKEN_HOST = 'https://oauth2.googleapis.com/token'
    DEFAULT_MTLS_TOKEN_HOST = 'https://oauth2.mtls.googleapis.com/token'

    def __init__(self):
        super(_SectionAuth, self).__init__('auth')
        self.auth_host = self._Add('auth_host', hidden=True, default=self.DEFAULT_AUTH_HOST)
        self.disable_credentials = self._AddBool('disable_credentials', default=False, help_text='If True, `gcloud` will not attempt to load any credentials or authenticate any requests. This is useful when behind a proxy that adds authentication to requests.')
        self.token_host = self._Add('token_host', default=self.DEFAULT_TOKEN_HOST, help_text='Overrides the token endpoint to provision access tokens. It can be used with Private Service Connect.')
        self.mtls_token_host = self._Add('mtls_token_host', default=self.DEFAULT_MTLS_TOKEN_HOST, help_text='Overrides the mtls token endpoint to provision access tokens.', hidden=True)
        self.disable_ssl_validation = self._AddBool('disable_ssl_validation', hidden=True)
        self.client_id = self._Add('client_id', hidden=True, default=config.CLOUDSDK_CLIENT_ID)
        self.client_secret = self._Add('client_secret', hidden=True, default=config.CLOUDSDK_CLIENT_NOTSOSECRET)
        self.authority_selector = self._Add('authority_selector', hidden=True)
        self.authorization_token_file = self._Add('authorization_token_file', hidden=True)
        self.credential_file_override = self._Add('credential_file_override', hidden=True)
        self.access_token_file = self._Add('access_token_file', help_text='A file path to read the access token. Use this property to authenticate gcloud with an access token. The credentials of the active account (if it exists) will be ignored. The file should contain an access token with no other information.')
        self.impersonate_service_account = self._Add('impersonate_service_account', help_text=textwrap.dedent("        While set, all API requests will be\n        made as the given service account or target service account in an\n        impersonation delegation chain instead of the currently selected\n        account. You can specify either a single service account as the\n        impersonator, or a comma-separated list of service accounts to\n        create an impersonation delegation chain. This is done without\n        needing to create, download, or activate a key for the service\n        account or accounts.\n        +\n        In order to make API requests as a service account, your\n        currently selected account must have an IAM role that includes\n        the `iam.serviceAccounts.getAccessToken` permission for the\n        service account or accounts.\n        +\n        The `roles/iam.serviceAccountTokenCreator` role has\n        the `iam.serviceAccounts.getAccessToken permission`. You can\n        also create a custom role.\n        +\n        You can specify a list of service accounts, separated with\n        commas. This creates an impersonation delegation chain in which\n        each service account delegates its permissions to the next\n        service account in the chain. Each service account in the list\n        must have the `roles/iam.serviceAccountTokenCreator` role on the\n        next service account in the list. For example, when the property is set\n        through `gcloud config set auth/impersonate_service_account=`\n        ``SERVICE_ACCOUNT_1'',``SERVICE_ACCOUNT_2'',\n        the active account must have the\n        `roles/iam.serviceAccountTokenCreator` role on\n        ``SERVICE_ACCOUNT_1'', which must have the\n        `roles/iam.serviceAccountTokenCreator` role on\n        ``SERVICE_ACCOUNT_2''.\n        ``SERVICE_ACCOUNT_1'' is the impersonated service\n        account and ``SERVICE_ACCOUNT_2'' is the delegate.\n        "))
        self.disable_code_verifier = self._AddBool('disable_code_verifier', default=False, hidden=True, help_text='Disable code verifier in 3LO auth flow. See https://tools.ietf.org/html/rfc7636 for more information about code verifier.')
        self.disable_load_google_auth = self._AddBool('disable_load_google_auth', default=False, hidden=True, help_text='Global switch to turn off loading credentials as google-auth. Users can use it to switch back to the old mode if google-auth breaks users.')
        self.opt_out_google_auth = self._AddBool('opt_out_google_auth', default=False, hidden=True, help_text='A switch to disable google-auth for a surface or a command group, in case there are some edge cases or google-auth does not work for some surface.')
        self.token_introspection_endpoint = self._Add('token_introspection_endpoint', hidden=True, help_text='Overrides the endpoint used for token introspection with Workload and Workforce Identity Federation. It can be used with Private Service Connect.')
        self.login_config_file = self._Add('login_config_file', help_text='Sets the created login configuration file in auth/login_config_file. Calling `gcloud auth login` will automatically use this login configuration unless it is explicitly unset.')
        self.service_account_use_self_signed_jwt = self._Add('service_account_use_self_signed_jwt', default=False, help_text='If True, use self signed jwt flow to get service account credentials access token. This only applies to service account json file and not to the legacy .p12 file.', validator=functools.partial(_BooleanValidator, 'service_account_use_self_signed_jwt'), choices=('true', 'false'))
        self.service_account_disable_id_token_refresh = self._AddBool('service_account_disable_id_token_refresh', default=False, help_text='If True, disable ID token refresh for service account.')
        self.reauth_use_google_auth = self._AddBool('reauth_use_google_auth', hidden=True, default=True, help_text='A switch to choose to use google-auth reauth or oauth2client reauth implementation. By default google-auth is used.')