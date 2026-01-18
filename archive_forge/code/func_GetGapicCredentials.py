from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account as google_auth_external_account
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
def GetGapicCredentials(enable_resource_quota=True, allow_account_impersonation=True):
    """Returns a credential object for use by gapic client libraries.

  Currently, we set _quota_project on the credentials, unlike for http requests,
  which add quota project through request wrapping to implement
  go/gcloud-quota-model-v2.

  Additionally, we wrap the refresh method and plug in our own
  google.auth.transport.Request object that uses our transport.

  Args:
    enable_resource_quota: bool, By default, we are going to tell APIs to use
        the quota of the project being operated on. For some APIs we want to use
        gcloud's quota, so you can explicitly disable that behavior by passing
        False here.
    allow_account_impersonation: bool, True to allow use of impersonated service
        account credentials for calls made with this client. If False, the
        active user credentials will always be used.

  Returns:
    A google auth credentials.Credentials object.

  Raises:
    MissingStoredCredentialsError: If a google-auth credential cannot be loaded.
  """
    credentials = store.LoadIfEnabled(allow_account_impersonation=allow_account_impersonation, use_google_auth=True)
    if not creds.IsGoogleAuthCredentials(credentials):
        raise MissingStoredCredentialsError('Unable to load credentials')
    if enable_resource_quota:
        credentials._quota_project_id = creds.GetQuotaProject(credentials)
    original_refresh = credentials.refresh

    def WrappedRefresh(request):
        del request
        if isinstance(credentials, google_auth_external_account.Credentials) and credentials.valid:
            return None
        return original_refresh(requests.GoogleAuthRequest())
    credentials.refresh = WrappedRefresh
    return credentials