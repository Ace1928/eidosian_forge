from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def GetSslCertificate(self, cert_id):
    """Gets a certificate for the given application.

    Args:
      cert_id: str, the id of the certificate to retrieve.

    Returns:
      The retrieved AuthorizedCertificate object.
    """
    request = self.messages.AppengineAppsAuthorizedCertificatesGetRequest(name=self._FormatSslCert(cert_id), view=self.messages.AppengineAppsAuthorizedCertificatesGetRequest.ViewValueValuesEnum.FULL_CERTIFICATE)
    return self.client.apps_authorizedCertificates.Get(request)