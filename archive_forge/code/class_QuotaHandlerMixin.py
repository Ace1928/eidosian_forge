from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.credentials import creds as core_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import files
from oauth2client import client
import six
from google.auth import exceptions as google_auth_exceptions
class QuotaHandlerMixin(object):
    """Mixin for handling quota project."""

    def QuotaProject(self, enable_resource_quota, allow_account_impersonation, use_google_auth):
        if not enable_resource_quota:
            return None
        creds = store.LoadIfEnabled(allow_account_impersonation, use_google_auth)
        return core_creds.GetQuotaProject(creds)

    def QuotaWrappedRequest(self, http_client, quota_project):
        """Returns a request method which adds the quota project header."""
        handlers = [transport.Handler(transport.SetHeader('X-Goog-User-Project', quota_project))]
        self.WrapRequest(http_client, handlers)
        return http_client.request

    @abc.abstractmethod
    def WrapQuota(self, http_client, enable_resource_quota, allow_account_impersonation, use_google_auth):
        """Returns a http_client with quota project handling.

    Args:
      http_client: The http client to be wrapped.
      enable_resource_quota: bool, By default, we are going to tell APIs to use
        the quota of the project being operated on. For some APIs we want to use
        gcloud's quota, so you can explicitly disable that behavior by passing
        False here.
      allow_account_impersonation: bool, True to allow use of impersonated
        service account credentials for calls made with this client. If False,
        the active user credentials will always be used.
      use_google_auth: bool, True if the calling command indicates to use
        google-auth library for authentication. If False, authentication will
        fallback to using the oauth2client library. If None, set the value based
        the configuration.
    """