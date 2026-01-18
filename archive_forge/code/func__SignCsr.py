from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import certificate_utils
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.privateca import create_utils
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import iam
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import p4sa
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.privateca import storage
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _SignCsr(self, issuer_pool_ref, csr, lifetime, issuer_ca_id):
    """Issues a certificate under the given issuer with the given settings."""
    certificate_id = 'subordinate-{}'.format(certificate_utils.GenerateCertId())
    issuer_pool_name = issuer_pool_ref.RelativeName()
    certificate_name = '{}/certificates/{}'.format(issuer_pool_name, certificate_id)
    cert_request = self.messages.PrivatecaProjectsLocationsCaPoolsCertificatesCreateRequest(certificateId=certificate_id, parent=issuer_pool_name, requestId=request_utils.GenerateRequestId(), issuingCertificateAuthorityId=issuer_ca_id, certificate=self.messages.Certificate(name=certificate_name, lifetime=lifetime, pemCsr=csr))
    return self.client.projects_locations_caPools_certificates.Create(cert_request)