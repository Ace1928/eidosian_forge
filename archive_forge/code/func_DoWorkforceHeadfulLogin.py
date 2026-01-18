from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account_authorized_user
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
def DoWorkforceHeadfulLogin(login_config_file, is_adc=False, **kwargs):
    """DoWorkforceHeadfulLogin attempts to log in with appropriate login configuration.

  It will return the account and credentials of the user if it succeeds

  Args:
    login_config_file (str): The path to the workforce headful login
      configuration file.
    is_adc (str): Whether the flow is initiated via application-default login.
    **kwargs (Mapping): Extra Arguments to pass to the method creating the flow.

  Returns:
    (google.auth.credentials.Credentials): The account and
    credentials of the user who logged in
  """
    login_config_data = auth_util.GetCredentialsConfigFromFile(login_config_file)
    if login_config_data.get('type', None) != 'external_account_authorized_user_login_config':
        raise calliope_exceptions.BadFileException('Only external account authorized user login config JSON credential file types are supported for Workforce Identity Federation login configurations.')
    client_config = _MakeThirdPartyClientConfig(login_config_data, is_adc)
    audience = login_config_data['audience']
    path_start = audience.find('/locations/')
    provider_name = None
    if path_start != -1:
        provider_name = audience[path_start + 1:]
    creds = auth_util.DoInstalledAppBrowserFlowGoogleAuth(config.CLOUDSDK_EXTERNAL_ACCOUNT_SCOPES, client_config=client_config, query_params={'provider_name': provider_name}, **kwargs)
    if isinstance(creds, external_account_authorized_user.Credentials):
        universe_domain_from_config = login_config_data.get('universe_domain', None)
        creds._universe_domain = universe_domain_from_config or properties.VALUES.core.universe_domain.Get()
    if not creds.audience:
        creds._audience = audience
    return creds