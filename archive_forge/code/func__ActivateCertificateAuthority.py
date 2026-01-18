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
def _ActivateCertificateAuthority(self, ca_name, pem_cert, issuer_chain):
    """Activates the given CA using the given certificate and issuing CA chain."""
    activate_request = self.messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesActivateRequest(name=ca_name, activateCertificateAuthorityRequest=self.messages.ActivateCertificateAuthorityRequest(pemCaCertificate=pem_cert, subordinateConfig=self.messages.SubordinateConfig(pemIssuerChain=self.messages.SubordinateConfigChain(pemCertificates=issuer_chain))))
    operation = self.client.projects_locations_caPools_certificateAuthorities.Activate(activate_request)
    return operations.Await(operation, 'Activating CA.', api_version='v1')