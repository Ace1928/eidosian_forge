from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def ListSslCertificates(self):
    """Lists all authorized certificates for the given application.

    Returns:
      A list of AuthorizedCertificate objects.
    """
    request = self.messages.AppengineAppsAuthorizedCertificatesListRequest(parent=self._FormatApp())
    response = self.client.apps_authorizedCertificates.List(request)
    return response.certificates