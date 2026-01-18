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
def GetElevationIdTokenGoogleAuth(self, google_auth_impersonation_credentials, audience, include_email):
    """Creates an ID token credentials for impersonated credentials."""
    from google.auth import impersonated_credentials as google_auth_impersonated_credentials
    from googlecloudsdk.core import requests as core_requests
    cred = google_auth_impersonated_credentials.IDTokenCredentials(google_auth_impersonation_credentials, target_audience=audience, include_email=include_email)
    request_client = core_requests.GoogleAuthRequest()
    self.PerformIamEndpointsOverride()
    cred.refresh(request_client)
    return cred