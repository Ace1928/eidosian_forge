from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def CreateSslCertificate(self, display_name, cert_path, private_key_path):
    """Creates a certificate for the given application.

    Args:
      display_name: str, the display name for the new certificate.
      cert_path: str, location on disk to a certificate file.
      private_key_path: str, location on disk to a private key file.

    Returns:
      The created AuthorizedCertificate object.

    Raises:
      Error if the file does not exist or can't be opened/read.
    """
    certificate_data = files.ReadFileContents(cert_path)
    private_key_data = files.ReadFileContents(private_key_path)
    cert = self.messages.CertificateRawData(privateKey=private_key_data, publicCertificate=certificate_data)
    auth_cert = self.messages.AuthorizedCertificate(displayName=display_name, certificateRawData=cert)
    request = self.messages.AppengineAppsAuthorizedCertificatesCreateRequest(parent=self._FormatApp(), authorizedCertificate=auth_cert)
    return self.client.apps_authorizedCertificates.Create(request)