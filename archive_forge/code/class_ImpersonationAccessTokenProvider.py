from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import transport
from oauth2client import client
class ImpersonationAccessTokenProvider(object):
    """A token provider for service account elevation.

  This supports the interface required by the core/credentials module.
  """

    def GetElevationAccessToken(self, service_account_id, scopes):
        if ',' in service_account_id:
            raise InvalidImpersonationAccount('More than one service accounts were specified, which is not supported. If being set, please unset the auth/disable_load_google_auth property and retry.')
        response = GenerateAccessToken(service_account_id, scopes)
        return ImpersonationCredentials(service_account_id, response.accessToken, response.expireTime, scopes)

    def GetElevationIdToken(self, service_account_id, audience, include_email):
        return GenerateIdToken(service_account_id, audience, include_email)

    def GetElevationAccessTokenGoogleAuth(self, source_credentials, target_principal, delegates, scopes):
        """Creates a fresh impersonation credential using google-auth library."""
        from google.auth import exceptions as google_auth_exceptions
        from google.auth import impersonated_credentials as google_auth_impersonated_credentials
        from googlecloudsdk.core import requests as core_requests
        request_client = core_requests.GoogleAuthRequest()
        source_credentials.refresh(request_client)
        cred = google_auth_impersonated_credentials.Credentials(source_credentials=source_credentials, target_principal=target_principal, target_scopes=scopes, delegates=delegates)
        self.PerformIamEndpointsOverride()
        try:
            cred.refresh(request_client)
        except google_auth_exceptions.RefreshError:
            raise ImpersonatedCredGoogleAuthRefreshError('Failed to impersonate [{service_acc}]. Make sure the account that\'s trying to impersonate it has access to the service account itself and the "roles/iam.serviceAccountTokenCreator" role.'.format(service_acc=target_principal))
        return cred

    def GetElevationIdTokenGoogleAuth(self, google_auth_impersonation_credentials, audience, include_email):
        """Creates an ID token credentials for impersonated credentials."""
        from google.auth import impersonated_credentials as google_auth_impersonated_credentials
        from googlecloudsdk.core import requests as core_requests
        cred = google_auth_impersonated_credentials.IDTokenCredentials(google_auth_impersonation_credentials, target_audience=audience, include_email=include_email)
        request_client = core_requests.GoogleAuthRequest()
        self.PerformIamEndpointsOverride()
        cred.refresh(request_client)
        return cred

    @classmethod
    def IsImpersonationCredential(cls, cred):
        from google.auth import impersonated_credentials as google_auth_impersonated_credentials
        return isinstance(cred, ImpersonationCredentials) or isinstance(cred, google_auth_impersonated_credentials.Credentials)

    @classmethod
    def PerformIamEndpointsOverride(cls):
        """Perform IAM endpoint override if needed.

    We will override IAM generateAccessToken, signBlob, and generateIdToken
    endpoint under the following conditions.
    (1) If the [api_endpoint_overrides/iamcredentials] property is explicitly
    set, we replace "https://iamcredentials.googleapis.com/" with the given
    property value in these endpoints.
    (2) If the property above is not set, and the [core/universe_domain] value
    is not default, we replace "googleapis.com" with the [core/universe_domain]
    property value in these endpoints.
    """
        from google.auth import impersonated_credentials as google_auth_impersonated_credentials
        effective_iam_endpoint = GetEffectiveIamEndpoint()
        google_auth_impersonated_credentials._IAM_ENDPOINT = google_auth_impersonated_credentials._IAM_ENDPOINT.replace(IAM_ENDPOINT_GDU, effective_iam_endpoint)
        google_auth_impersonated_credentials._IAM_SIGN_ENDPOINT = google_auth_impersonated_credentials._IAM_SIGN_ENDPOINT.replace(IAM_ENDPOINT_GDU, effective_iam_endpoint)
        google_auth_impersonated_credentials._IAM_IDTOKEN_ENDPOINT = google_auth_impersonated_credentials._IAM_IDTOKEN_ENDPOINT.replace(IAM_ENDPOINT_GDU, effective_iam_endpoint)